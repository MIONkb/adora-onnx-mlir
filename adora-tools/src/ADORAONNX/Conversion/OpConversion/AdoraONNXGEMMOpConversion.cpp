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

// TODO: Replace with the actual ADORA tensor dialect header in your project.
// For example:
//   #include "ADORA/Dialect/ADORATensor/IR/ADORATensorOps.h"
// or:
//   #include "src/Dialect/ADORA/IR/ADORATensorOps.hpp"
//
// The ideal case is to create GEMMOp via its generated C++ class:
//   mlir::ADORA::ADORATensor::GEMMOp
//
// If the C++ op class is not available, this utility falls back to creating
// the operation by opname string (e.g. "adora.GEMM") via OperationState.

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

/// Lowers a single onnx.Gemm op into an ADORA GEMM op.
///
/// ONNX Gemm carries alpha/beta/transA/transB attributes. This routine forwards
/// them to the target op, and replaces all uses of the original result.
static LogicalResult lowerONNXGemmToAdoraGemm(onnx::ONNXGemmOp gemm,
                                             OpBuilder &builder) {
  Location loc = gemm.getLoc();
  Value A = gemm.getA();
  Value B = gemm.getB();
  Value C = gemm.getC();

  // ONNX Gemm attributes: alpha, beta, transA, transB.
  auto alphaAttr = gemm.getAlphaAttr();
  auto betaAttr = gemm.getBetaAttr();
  auto transAAttr = gemm.getTransAAttr();
  auto transBAttr = gemm.getTransBAttr();

  Type outTy = gemm.getY().getType();

  builder.setInsertionPoint(gemm);

  // Preferred form (if you have the generated op class):
  //
  // auto newOp = builder.create<mlir::ADORA::ADORATensor::GEMMOp>(
  //     loc, outTy, A, B, C,
  //     /*alpha=*/alphaAttr, /*beta=*/betaAttr,
  //     /*transA=*/transAAttr, /*transB=*/transBAttr);

  // Fallback: create by opname string. Replace "adora.GEMM" with the actual
  // TableGen op name used in your dialect (e.g. "adora.gemm").
  OperationState st(loc, "adora.GEMM");
  st.addOperands({A, B, C});
  st.addTypes({outTy});
  if (alphaAttr)
    st.addAttribute("alpha", alphaAttr);
  if (betaAttr)
    st.addAttribute("beta", betaAttr);
  if (transAAttr)
    st.addAttribute("transA", transAAttr);
  if (transBAttr)
    st.addAttribute("transB", transBAttr);
  Operation *newOp = builder.create(st);

  Value newY = newOp->getResult(0);
  gemm.getY().replaceAllUsesWith(newY);
  gemm.erase();

  return success();
}

//===----------------------------------------------------------------------===//
// Lowering: onnx.MatMul (+ optional onnx.Add) -> adora.GEMM
//===----------------------------------------------------------------------===//

/// Lowers MatMul-like patterns into an ADORA GEMM op.
///
/// - If `biasOrNull` is provided, this performs fusion equivalent to:
///     Y = A * B + bias
///   and sets beta = 1.
///
/// - If `biasOrNull` is null, this lowers plain MatMul:
///     Y = A * B
///   and sets beta = 0.
///
/// `anchorToErase` is the MatMul op to remove.
/// `optionalAddToErase` is the Add op to remove (if fused).
static LogicalResult lowerMatMulLikeToAdoraGemm(Value A, Value B,
                                               Value biasOrNull, Type outTy,
                                               Location loc,
                                               Operation *anchorToErase,
                                               Operation *optionalAddToErase,
                                               OpBuilder &builder) {
  MLIRContext *ctx = builder.getContext();
  builder.setInsertionPoint(anchorToErase);

  // MatMul defaults: alpha=1, transA=0, transB=0.
  // If bias exists, set beta=1; otherwise beta=0.
  Attribute alphaAttr = f32Attr(ctx, 1.0f);
  Attribute betaAttr = f32Attr(ctx, biasOrNull ? 1.0f : 0.0f);

  // Use i64 0/1 flags for transpose attributes (consistent with ONNX Gemm).
  auto i64Ty = IntegerType::get(ctx, 64);
  IntegerAttr transAAttr = IntegerAttr::get(i64Ty, 0);
  IntegerAttr transBAttr = IntegerAttr::get(i64Ty, 0);

  // Note: ONNX Gemm formally has a third input C. For MatMul lowering, we only
  // pass C if we fused a bias(Add). If your GEMMOp requires a C operand even
  // for beta=0, you may need to materialize a zero tensor here.

  // Preferred form (if you have the generated op class):
  //
  // auto newOp = builder.create<mlir::ADORA::ADORATensor::GEMMOp>(
  //     loc, outTy, A, B, /*C=*/biasOrNull,
  //     /*alpha=*/alphaAttr, /*beta=*/betaAttr,
  //     /*transA=*/transAAttr, /*transB=*/transBAttr);

  OperationState st(loc, "adora.GEMM");
  if (biasOrNull)
    st.addOperands({A, B, biasOrNull});
  else
    st.addOperands({A, B});
  st.addTypes({outTy});
  st.addAttribute("alpha", alphaAttr);
  st.addAttribute("beta", betaAttr);
  st.addAttribute("transA", transAAttr);
  st.addAttribute("transB", transBAttr);

  Operation *newOp = builder.create(st);
  Value newY = newOp->getResult(0);

  // Replace users: if we fused Add, replace Add's result; otherwise replace
  // MatMul's result.
  if (optionalAddToErase) {
    optionalAddToErase->getResult(0).replaceAllUsesWith(newY);
  } else {
    anchorToErase->getResult(0).replaceAllUsesWith(newY);
  }

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
  llvm::SmallVector<onnx::ONNXAddOp, 16> addOps;
  module.walk([&](onnx::ONNXAddOp add) { addOps.push_back(add); });

  for (auto add : addOps) {
    if (!add || add->isErased())
      continue;

    Value lhs = add.getA();
    Value rhs = add.getB();

    Operation *lhsDef = lhs.getDefiningOp();
    Operation *rhsDef = rhs.getDefiningOp();

    onnx::ONNXMatMulOp mm = nullptr;
    Value mmRes;
    Value bias;

    // Match either:
    //   Add(MatMul(...), bias)
    // or:
    //   Add(bias, MatMul(...))
    if (lhsDef && (mm = dyn_cast<onnx::ONNXMatMulOp>(lhsDef))) {
      mmRes = lhs;
      bias = rhs;
    } else if (rhsDef && (mm = dyn_cast<onnx::ONNXMatMulOp>(rhsDef))) {
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
  llvm::SmallVector<onnx::ONNXMatMulOp, 16> mmOps;
  module.walk([&](onnx::ONNXMatMulOp mm) { mmOps.push_back(mm); });

  for (auto mm : mmOps) {
    if (!mm || mm->isErased())
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
  llvm::SmallVector<onnx::ONNXGemmOp, 16> gemmOps;
  module.walk([&](onnx::ONNXGemmOp g) { gemmOps.push_back(g); });

  for (auto g : gemmOps) {
    if (!g || g->isErased())
      continue;
    (void)lowerONNXGemmToAdoraGemm(g, builder);
  }
}
}
} // namespace
} // namespace