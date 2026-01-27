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

#define DEBUG_TYPE "onnx-to-adora"

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


static Value getMemRefDimValue(OpBuilder &builder, Location loc, Value base,
                               int64_t dim) {
  auto baseType = base.getType().cast<MemRefType>();
  if (!baseType.isDynamicDim(dim))
    return builder.create<arith::ConstantIndexOp>(loc, baseType.getDimSize(dim));
  return builder.create<memref::DimOp>(loc, base, dim);
}

static Value createRankReducedSubview(OpBuilder &b, Location loc,
                                      Value base,
                                      ArrayRef<int64_t> loopDims,
                                      ArrayRef<Value> loopIvs) {
  auto baseTy = base.getType().cast<MemRefType>();
  int64_t rank = baseTy.getRank();
  assert(loopDims.size() == loopIvs.size());

  // Map ivs to dims
  SmallVector<Value, 4> ivForDim(rank, Value());
  for (size_t i = 0; i < loopDims.size(); ++i)
    ivForDim[loopDims[i]] = loopIvs[i];

  SmallVector<OpFoldResult, 4> offsets, sizes, strides;
  offsets.reserve(rank);
  sizes.reserve(rank);
  strides.reserve(rank);

  auto idx0 = b.getIndexAttr(0);
  auto idx1 = b.getIndexAttr(1);

  for (int64_t dim = 0; dim < rank; ++dim) {
    strides.push_back(idx1);

    if (dim < rank - 2) {
      // reduce-away dims: offset = iv or 0, size = 1
      offsets.push_back(ivForDim[dim] ? OpFoldResult(ivForDim[dim]) : OpFoldResult(idx0));
      sizes.push_back(idx1);
    } else {
      // keep last 2 dims: offset = 0, size = full dim (dynamic or static)
      offsets.push_back(idx0);
      if (baseTy.isDynamicDim(dim))
        sizes.push_back(getMemRefDimValue(b, loc, base, dim));   // Value (index)
      else
        sizes.push_back(b.getIndexAttr(baseTy.getDimSize(dim))); // Attr
    }
  }

  // resultShape: dynamic kept dims become kDynamic (-1)
  SmallVector<int64_t, 2> resultShape = {
      baseTy.isDynamicDim(rank - 2) ? ShapedType::kDynamic : baseTy.getDimSize(rank - 2),
      baseTy.isDynamicDim(rank - 1) ? ShapedType::kDynamic : baseTy.getDimSize(rank - 1),
  };
  baseTy.dump();
  for(auto _ : offsets){
    _.dump();
  }
  for(auto _ : sizes){
    _.dump();
  }
  for(auto _ : strides){
    _.dump();
  }
  auto reducedTy =
      memref::SubViewOp::inferRankReducedResultType(resultShape, baseTy,
                                                    offsets, sizes, strides)
          .cast<MemRefType>();

  // IMPORTANT: build with OFR so dynamic sizes/offsets are carried.
  return b.create<memref::SubViewOp>(loc, reducedTy, base, offsets, sizes, strides);
}

static void lowerBatchedADORAGemmOp(mlir::ADORA::ADORATensor::GemmOp op) {
  LLVM_DEBUG(llvm::errs() << "lowerBatchedADORAGemmOp: "<< op;);
  auto aType = op.getA().getType().dyn_cast<MemRefType>();
  auto bType = op.getB().getType().dyn_cast<MemRefType>();
  auto cType = op.getC().getType().dyn_cast<MemRefType>();
  auto outType = op.getO().getType().dyn_cast<MemRefType>();
  if (!aType || !bType || !cType || !outType)
    return;
  if (aType.getRank() <= 2)
    return;

  int64_t rank = aType.getRank();
  SmallVector<int64_t, 4> loopDims;
  loopDims.reserve(rank - 2);
  for (int64_t dim = 0; dim < rank - 2; ++dim) {
    if (!aType.isDynamicDim(dim) && aType.getDimSize(dim) == 1)
      continue;
    loopDims.push_back(dim);
  }

  OpBuilder builder(op);
  Location loc = op.getLoc();

  SmallVector<Value, 4> outDynSizes;
  for (int64_t dim = 0; dim < rank; ++dim)
    if (outType.isDynamicDim(dim))
      outDynSizes.push_back(getMemRefDimValue(builder, loc, op.getC(), dim));

  Value outAlloc =
      builder.create<memref::AllocOp>(loc, outType, outDynSizes);

  SmallVector<Value, 4> loopIvs;
  auto emitGemmSlice = [&](OpBuilder &nestedBuilder) {
    Value aSubview = createRankReducedSubview(
        nestedBuilder, loc, op.getA(), loopDims, loopIvs);
    Value bSubview = createRankReducedSubview(
        nestedBuilder, loc, op.getB(), loopDims, loopIvs);
    Value cSubview = createRankReducedSubview(
        nestedBuilder, loc, op.getC(), loopDims, loopIvs);
    Value outSubview = createRankReducedSubview(
        nestedBuilder, loc, outAlloc, loopDims, loopIvs);

    auto gemm2d = nestedBuilder.create<mlir::ADORA::ADORATensor::GemmOp>(
        loc, outSubview.getType(), aSubview, bSubview, cSubview);
    nestedBuilder.create<memref::CopyOp>(loc, gemm2d.getO(), outSubview);
  };

  std::function<void(size_t, OpBuilder &)> buildLoopNest =
      [&](size_t idx, OpBuilder &nestBuilder) {
        if (idx == loopDims.size()) {
          emitGemmSlice(nestBuilder);
          return;
        }
        int64_t dim = loopDims[idx];
        Value upper = getMemRefDimValue(nestBuilder, loc, op.getA(), dim);
        auto ubMap =
            AffineMap::get(0, 1, nestBuilder.getAffineSymbolExpr(0));
        auto lbMap = AffineMap::getConstantMap(0, nestBuilder.getContext());
        auto forOp = nestBuilder.create<affine::AffineForOp>(
            loc,
            /*lbOperands=*/ValueRange{},
            /*lbMap=*/lbMap,
            /*ubOperands=*/ValueRange{upper},
            /*ubMap=*/ubMap,
            /*step=*/1);
        OpBuilder bodyBuilder = OpBuilder::atBlockBegin(forOp.getBody());
        loopIvs.push_back(forOp.getInductionVar());
        buildLoopNest(idx + 1, bodyBuilder);
        loopIvs.pop_back();
      };

  if (loopDims.empty()) {
    emitGemmSlice(builder);
  } else {
    buildLoopNest(0, builder);
  }

  op.replaceAllUsesWith(outAlloc);
  op.erase();
}

void lowerBatchedADORAGemmOps(ModuleOp module) {
  SmallVector<mlir::ADORA::ADORATensor::GemmOp, 8> gemmOps;
  module.walk([&](mlir::ADORA::ADORATensor::GemmOp op) { gemmOps.push_back(op); });
  for (auto op : gemmOps)
    if (op)
      lowerBatchedADORAGemmOp(op);
}

}
} // namespace
} // namespace