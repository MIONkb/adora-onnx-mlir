/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===- FuseONNXOperatorToAdoraTensor.cpp - Fuse ONNX ops into ADORA tensor ops ---===//
//
// This file provides a lightweight IR rewriting utility (not a pass and not a
// RewritePattern) that fuses selected ONNX ops into a single ADORA tensor op.
//
// Currently supported conversions:
//   - onnx.Gemm   -> adora.GEMM
//   - onnx.MatMul -> adora.GEMM
//   - onnx.MatMul + onnx.Add (compatible bias) -> adora.GEMM (fused)
//
// The public entry point is:
//   void FuseONNXOperatorToAdoraTensor(ModuleOp module);
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
// #include "mlir/IR/BlockAndValueMapping.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "ADORA/Dialect/ADORATensor/IR/ADORATensor.h"

using namespace mlir;
using namespace onnx_mlir;
namespace mlir {
namespace ADORA {
namespace ADORATensor {
//===----------------------------------------------------------------------===//
// Shape / type helpers
//===----------------------------------------------------------------------===//

/// Returns true if `v` has a RankedTensorType.
static bool isRankedTensor(Value v) {
  return v.getType().isa<RankedTensorType>();
}

/// Returns true if `a` and `b` are tensor types with the same element type.
static bool sameElementType(Type a, Type b) {
  auto ta = a.dyn_cast<TensorType>();
  auto tb = b.dyn_cast<TensorType>();
  if (!ta || !tb)
    return false;
  return ta.getElementType() == tb.getElementType();
}

/// A conservative broadcast compatibility check.
///
/// - For fully static shapes: allow NumPy/ONNX-style trailing-dimension broadcast
///   (aligned from the last dimension; dim == 1 or dim == targetDim).
/// - For dynamic dims: conservatively accept (could be tightened later).
static bool isBroadcastableTo(RankedTensorType from, RankedTensorType to) {
  if (from.getElementType() != to.getElementType())
    return false;

  int64_t rFrom = from.getRank();
  int64_t rTo = to.getRank();
  if (rFrom > rTo)
    return false;

  auto fromShape = from.getShape();
  auto toShape = to.getShape();

  // Compare from trailing dims.
  for (int64_t i = 0; i < rTo; ++i) {
    int64_t toDim = toShape[rTo - 1 - i];
    int64_t fromDim = (i < rFrom) ? fromShape[rFrom - 1 - i] : 1;

    // If any dim is dynamic, accept conservatively.
    if (ShapedType::isDynamic(toDim) || ShapedType::isDynamic(fromDim))
      continue;

    if (fromDim != 1 && fromDim != toDim)
      return false;
  }
  return true;
}

/// Returns true if `bias` can legally serve as the additive term for result `y`.
/// Accept either:
///   - exact shape match
///   - broadcastable-to-y
static bool isBiasCompatible(Value bias, Value y) {
  auto yTy = y.getType().dyn_cast<RankedTensorType>();
  auto bTy = bias.getType().dyn_cast<RankedTensorType>();
  if (!yTy || !bTy)
    return false;
  if (!sameElementType(bTy, yTy))
    return false;

  if (bTy == yTy)
    return true;
  return isBroadcastableTo(bTy, yTy);
}

//===----------------------------------------------------------------------===//
// Attribute helpers
//===----------------------------------------------------------------------===//

/// Convenience wrapper for creating f32 FloatAttr constants.
static Attribute f32Attr(MLIRContext *ctx, float v) {
  return FloatAttr::get(Float32Type::get(ctx), v);
}

/// Convenience wrapper for creating f64 FloatAttr constants.
static Attribute f64Attr(MLIRContext *ctx, double v) {
  return FloatAttr::get(Float64Type::get(ctx), v);
}

/// Note: If GEMMOp expects alpha/beta as SSA Values rather than Attributes,
/// replace these FloatAttr builders with arith.constant materialization.

//===----------------------------------------------------------------------===//
// Lowering: onnx.Gemm -> adora.GEMM
//===----------------------------------------------------------------------===//

/// Lowers a single onnx.Gemm op into an ADORATensor::GemmOp.
///
/// This lowering only maps operands (A, B, C) and the result type.
/// ONNX attributes (alpha/beta/transA/transB) are ignored because the target
/// ADORATensor::GemmOp currently carries no such attributes.
static LogicalResult lowerONNXGemmToAdoraGemm(mlir::ONNXGemmOp gemm,
                                             OpBuilder &builder) {
  Location loc = gemm.getLoc();

  Value A = gemm.getA();
  Value B = gemm.getB();
  Value C = gemm.getC();

  Type outTy = gemm.getY().getType();

  builder.setInsertionPoint(gemm);

  // Create ADORATensor.Gemm with exactly 3 operands and 1 result.
  auto newOp = builder.create<mlir::ADORA::ADORATensor::GemmOp>(
      loc, outTy, A, B, C);

  Value newY = newOp.getO();
  gemm.getY().replaceAllUsesWith(newY);
  gemm.erase();

  return success();
}

//===----------------------------------------------------------------------===//
// Lowering: onnx.MatMul (+ optional onnx.Add) -> ADORATensor.Gemm
//===----------------------------------------------------------------------===//

/// Lowers MatMul-like patterns into ADORATensor::GemmOp.
///
/// This lowering assumes ADORATensor::GemmOp has the fixed semantics:
///   O = A * B + C
/// with exactly three operands (A, B, C) and no extra attributes.
///
/// Therefore:
///   - If `biasOrNull` is provided, we fuse:
///       Add(MatMul(A, B), bias)  ==>  Gemm(A, B, bias)
///   - If `biasOrNull` is null, lowering is rejected because a valid `C` operand
///     is required by the target op.
///
/// `anchorToErase` is the MatMul op to remove.
/// `optionalAddToErase` is the Add op to remove (if fused).
static LogicalResult lowerMatMulLikeToAdoraGemm(Value A, Value B,
                                               Value biasOrNull, Type outTy,
                                               Location loc,
                                               Operation *anchorToErase,
                                               Operation *optionalAddToErase,
                                               OpBuilder &builder) {
  builder.setInsertionPoint(anchorToErase);

  // ADORATensor::GemmOp requires a third operand C. If we did not fuse a bias,
  // we cannot form a valid target op without materializing a zero tensor.
  if (!biasOrNull)
    return failure();

  // Create ADORATensor.Gemm with exactly 3 operands and 1 result.
  auto newOp = builder.create<mlir::ADORA::ADORATensor::GemmOp>(
      loc, outTy, A, B, /*C=*/biasOrNull);

  Value newY = newOp.getO();

  // Replace users: if we fused Add, replace Add's result; otherwise replace
  // MatMul's result.
  if (optionalAddToErase)
    optionalAddToErase->getResult(0).replaceAllUsesWith(newY);
  else
    anchorToErase->getResult(0).replaceAllUsesWith(newY);

  // Erase in dependency order: erase Add first, then MatMul.
  if (optionalAddToErase)
    optionalAddToErase->erase();
  anchorToErase->erase();

  return success();
}

//===----------------------------------------------------------------------===//
// Public entry point
//===----------------------------------------------------------------------===//

/// Top-level driver that performs local fusion/lowering within a ModuleOp.
///
/// The routine applies transformations in the following order:
///  1) Fuse MatMul + Add patterns first (they delete MatMul).
///  2) Lower remaining MatMul ops.
///  3) Lower Gemm ops.
void FuseONNXOperatorToAdoraTensor(ModuleOp module) {
  MLIRContext *ctx = module.getContext();
  OpBuilder builder(ctx);

  // Step 1: Fuse MatMul + Add first, because this removes the MatMul op as well.
  llvm::SmallVector<mlir::ONNXAddOp, 16> addOps;
  module.walk([&](mlir::ONNXAddOp add) { addOps.push_back(add); });

  for (auto add : addOps) {
    // if (!add || add->isErased())
    if (!add)
      continue;

    Value lhs = add.getA();
    Value rhs = add.getB();

    Operation *lhsDef = lhs.getDefiningOp();
    Operation *rhsDef = rhs.getDefiningOp();

    mlir::ONNXMatMulOp mm = nullptr;
    Value mmRes;
    Value bias;

    // Match either:
    //   Add(MatMul(...), bias)
    // or:
    //   Add(bias, MatMul(...))
    if (lhsDef && (mm = dyn_cast<mlir::ONNXMatMulOp>(lhsDef))) {
      mmRes = lhs;
      bias = rhs;
    } else if (rhsDef && (mm = dyn_cast<mlir::ONNXMatMulOp>(rhsDef))) {
      mmRes = rhs;
      bias = lhs;
    } else {
      continue;
    }

    // Require MatMul result to have a single use (the Add), otherwise fusing
    // would change semantics for other users.
    if (!mmRes.hasOneUse())
      continue;

    // Conservative type/shape checks.
    if (!isRankedTensor(mmRes) || !isRankedTensor(bias) ||
        !isRankedTensor(add.getResult()))
      continue;

    // Ensure the additive term is shape-compatible with the MatMul result.
    if (!isBiasCompatible(bias, add.getResult()))
      continue;

    Value A = mm.getA();
    Value B = mm.getB();
    Type outTy = add.getResult().getType();
    Location loc = add.getLoc();

    (void)lowerMatMulLikeToAdoraGemm(
        A, B, bias, outTy, loc,
        /*anchorToErase=*/mm.getOperation(),
        /*optionalAddToErase=*/add.getOperation(), builder);
  }

  // Step 2: Lower remaining MatMul ops (without bias fusion).
  llvm::SmallVector<mlir::ONNXMatMulOp, 16> mmOps;
  module.walk([&](mlir::ONNXMatMulOp mm) { mmOps.push_back(mm); });

  for (auto mm : mmOps) {
    // if (!mm || mm->isErased())
    if (!mm)
      continue;

    Value Y = mm.getResult();
    if (!isRankedTensor(Y))
      continue;

    Value A = mm.getA();
    Value B = mm.getB();
    Type outTy = Y.getType();
    Location loc = mm.getLoc();

    (void)lowerMatMulLikeToAdoraGemm(
        A, B, /*biasOrNull=*/Value(), outTy, loc,
        /*anchorToErase=*/mm.getOperation(),
        /*optionalAddToErase=*/nullptr, builder);
  }

  // Step 3: Lower onnx.Gemm ops directly.
  llvm::SmallVector<mlir::ONNXGemmOp, 16> gemmOps;
  module.walk([&](mlir::ONNXGemmOp g) { gemmOps.push_back(g); });

  for (auto g : gemmOps) {
    // if (!g || g->isErased())
    if (!g)
      continue;
    (void)lowerONNXGemmToAdoraGemm(g, builder);
  }
}
}
} // namespace
} // namespace