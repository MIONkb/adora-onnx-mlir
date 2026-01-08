//===- LinalgToSystolicGEMM.cpp - conversion from Linalg named operators to systolic gemm --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// #include "mlir/Dialect/Linalg/Passes.h"
#include <functional>

#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Utils/AffineCanonicalizationUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"

#include "ADORA/Dialect/ADORA/IR/ADORA.h"
#include "ADORA/Dialect/ADORA/IR/KernelOp/ADORAKernelOp.h"
#include "ADORA/Dialect/ADORATensor/IR/ADORATensor.h"
#include "ADORA/Dialect/ADORATensor/Utility/Utility.h"
// #include "ADORA/Dialect/ADORATensor/Transforms/Passes.h"
#include "ADORAONNX/Conversion/Passes.h"
#include "ADORAONNX/Conversion/ONNXToADORACommon.hpp"

#include "src/Dialect/ONNX/ONNXDialect.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Dialect/Krnl/DialectBuilder.hpp"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
// #include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
// #include "src/Dialect/Mlir/VectorMachineSupport.hpp"
// #include "PassDetail.h"

#define DEBUG_TYPE "onnx-to-adora"

namespace mlir {
#include "mlir/Dialect/Linalg/Passes.h.inc"
} // namespace mlir

using namespace llvm;
using namespace mlir;
using namespace mlir::linalg;
using namespace mlir::ADORA;
using namespace mlir::ADORA::ADORATensor;
using namespace onnx_mlir;

namespace mlir {
namespace ADORA {
namespace ADORATensor {

void markElementWiseOpDynamicIllegal(ConversionTarget& target){
  target.addDynamicallyLegalOp<ONNXMulOp>([&](ONNXMulOp op) {
    return !isLargeTensor(op.getA()) && !isLargeTensor(op.getB());
  });
  target.addDynamicallyLegalOp<ONNXAddOp>([&](ONNXAddOp op) {
    return !isLargeTensor(op.getA()) && !isLargeTensor(op.getB());
  });
}

static Value getMemRefDimValue(OpBuilder &builder, Location loc, Value base,
                               int64_t dim) {
  auto baseType = base.getType().cast<MemRefType>();
  if (!baseType.isDynamicDim(dim))
    return builder.create<arith::ConstantIndexOp>(loc, baseType.getDimSize(dim));
  return builder.create<memref::DimOp>(loc, base, dim);
}

static Value createRankReducedSubview(OpBuilder &builder, Location loc,
                                      Value base,
                                      ArrayRef<int64_t> loopDims,
                                      ArrayRef<Value> loopIvs) {
  auto baseType = base.getType().cast<MemRefType>();
  int64_t rank = baseType.getRank();

  SmallVector<Value, 4> ivForDim(rank, Value());
  for (size_t i = 0; i < loopDims.size(); ++i)
    ivForDim[loopDims[i]] = loopIvs[i];

  SmallVector<OpFoldResult, 4> offsets;
  SmallVector<OpFoldResult, 4> sizes;
  SmallVector<OpFoldResult, 4> strides;
  offsets.reserve(rank);
  sizes.reserve(rank);
  strides.reserve(rank);

  for (int64_t dim = 0; dim < rank; ++dim) {
    if (dim < rank - 2) {
      if (ivForDim[dim])
        offsets.push_back(ivForDim[dim]);
      else
        offsets.push_back(builder.getIndexAttr(0));
      sizes.push_back(builder.getIndexAttr(1));
    } else {
      offsets.push_back(builder.getIndexAttr(0));
      if (baseType.isDynamicDim(dim))
        sizes.push_back(getMemRefDimValue(builder, loc, base, dim));
      else
        sizes.push_back(builder.getIndexAttr(baseType.getDimSize(dim)));
    }
    strides.push_back(builder.getIndexAttr(1));
  }

  auto reducedType = memref::SubViewOp::inferRankReducedResultType(
      2, baseType, offsets, sizes, strides);
  return builder.create<memref::SubViewOp>(loc, reducedType, base, offsets, sizes,
                                          strides);
}

static void lowerBatchedADORAGemmOp(mlir::ADORA::ADORATensor::GemmOp op) {
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
            loc, lbMap, ValueRange{}, ubMap, ValueRange{upper}, 1);
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

static void lowerBatchedADORAGemmOps(ModuleOp module) {
  SmallVector<mlir::ADORA::ADORATensor::GemmOp, 8> gemmOps;
  module.walk([&](mlir::ADORA::ADORATensor::GemmOp op) { gemmOps.push_back(op); });
  for (auto op : gemmOps)
    if (op)
      lowerBatchedADORAGemmOp(op);
}

/// Lowering pattern: remove onnx.entry op and mark the target func as entry.
struct EraseONNXEntryPointPattern
    : public OpRewritePattern<ONNXEntryPointOp> {
  using OpRewritePattern<ONNXEntryPointOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXEntryPointOp op, PatternRewriter &rewriter) const override {
    ModuleOp module = op.getOperation()->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    SymbolRefAttr funcRefAttr = op->getAttrOfType<SymbolRefAttr>(
        ONNXEntryPointOp::getEntryPointFuncAttrName());
    if (!funcRefAttr)
      return op.emitError("ONNXEntryPointOp missing entry function symbol");

    StringRef entryPointName = funcRefAttr.getLeafReference().getValue();

    Operation *entryPointOp = module.lookupSymbol(entryPointName);
    if (!entryPointOp)
      return op.emitError() << "entry point function '" << entryPointName
                            << "' not found in module";

    func::FuncOp entryPointFunc = mlir::cast<func::FuncOp>(entryPointOp);

    entryPointFunc->setAttr(
        "onnxEntryPoint", rewriter.getUnitAttr());

    rewriter.eraseOp(op);
    return success();
  }
};


/// Generic type-only conversion pattern for ADORATensor ops.
/// This pattern rewrites ADORATensor operations by converting
/// their tensor-typed results into target-legal types (e.g., memref),
/// while preserving the original operation semantics.
///
/// The pattern:
///   1) Converts all result types using the provided TypeConverter.
///   2) Recreates the same ADORATensor op with converted result types
///      and already-converted operands.
///   3) Moves regions if the op contains any (for future extensibility).
///   4) Replaces the original op with the new one.
template <typename ADORATENSOROP_TYPE>
struct ADORATensorTypeConversion
    : OpConversionPattern<ADORATENSOROP_TYPE> {

  // using OpConversionPattern<ADORATENSOROP_TYPE>::OpConversionPattern;

  ADORATensorTypeConversion(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern<ADORATENSOROP_TYPE>(typeConverter, ctx) {};

  LogicalResult matchAndRewrite(
      ADORATENSOROP_TYPE op,
      typename ADORATENSOROP_TYPE::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    // 1. Convert all result types (support multi-result)
    SmallVector<Type, 4> newResultTypes;
    if (failed(this->getTypeConverter()->convertTypes(
            op->getResultTypes(), newResultTypes)))
      return failure();

    // 2. Clone op with new types + converted operands
    auto newOp = rewriter.create<ADORATENSOROP_TYPE>(
        op.getLoc(),
        newResultTypes,
        adaptor.getOperands(),
        op->getAttrs());

    // // 3. Move regions if any (future-proof)
    // rewriter.inlineRegionBefore(
    //     op->getRegions(), newOp->getRegions(),
    //     newOp->getRegions().begin());

    // 4. Replace
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

void populateONNXToKrnlConversionPatternInAdora(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx, DimAnalysis *dimAnalysis,
    bool enableTiling = false, bool enableSIMD = false, bool enableParallel = false,
    bool enableFastMath = false) {
  populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns, typeConverter);
  populateCallOpTypeConversionPattern(patterns, typeConverter);
  populateReturnOpTypeConversionPattern(patterns, typeConverter);

  // Math  
  populateLoweringONNXElementwiseOpPattern(patterns, typeConverter, ctx, dimAnalysis, enableSIMD, enableParallel);
  populateLoweringONNXHardmaxOpPattern(patterns, typeConverter, ctx);
  populateLoweringONNXReductionOpPattern(patterns, typeConverter, ctx, enableSIMD, enableParallel);  
  populateLoweringONNXSoftmaxOpPattern(patterns, typeConverter, ctx, enableParallel);
  populateLoweringONNXTopKOpPattern(patterns, typeConverter, ctx);
  populateLoweringONNXMatMulOpPattern(patterns, typeConverter, ctx, dimAnalysis, enableTiling, enableSIMD, enableParallel);

  // Tensor
  populateLoweringONNXDimOpPattern(patterns, typeConverter, ctx);
  populateLoweringONNXReshapeOpPattern(patterns, typeConverter, ctx, dimAnalysis);
  // populateLoweringONNXTransposeOpPattern(patterns, typeConverter, ctx, enableParallel); ## customized lower in adora
  populateLoweringONNXConstantOfShapeOpPattern(patterns, typeConverter, ctx);
  populateLoweringONNXConstantOpPattern(patterns, typeConverter, ctx);
  populateLoweringONNXConcatOpPattern(patterns, typeConverter, ctx, enableParallel);
  populateLoweringONNXSliceOpPattern(patterns, typeConverter, ctx);

  // Neural network
  populateLoweringONNXNormalizationOpPattern(patterns, typeConverter, ctx, dimAnalysis, enableSIMD, enableParallel);

  // populateLoweringONNXEntryPoint(patterns, ctx);
  populateAdoraLoweringONNXTransposeOpPattern(patterns, typeConverter, ctx);
  patterns.insert<EraseONNXEntryPointPattern>(ctx);

  // adora tensor op type conversion
  patterns.insert<ADORATensorTypeConversion<mlir::ADORA::ADORATensor::GemmOp>>(typeConverter, ctx);
  patterns.insert<ADORATensorTypeConversion<mlir::ADORA::ADORATensor::MatMulOp>>(typeConverter, ctx);
}

struct ConvertONNXLayersToAdoraPass
    : public ConvertONNXLayersToAdoraPassBase<ConvertONNXLayersToAdoraPass> {
public:
  int matmul_idx = 0;
  mlir::ModuleOp _m;

  void runOnOperation();
  // func::FuncOp ConvertMatmulToSystolic(linalg::MatmulOp matmul);
}; // struct OutlineONNXLayersPass



void ConvertONNXLayersToAdoraPass::runOnOperation()  {
  ModuleOp module = getOperation();
  DimAnalysis *dimAnalysis = new DimAnalysis(module);
  dimAnalysis->analyze();
  
  // The first thing to define is the conversion target. This will define the
  // final target for this lowering.
  ConversionTarget target(getContext());

  target.addLegalDialect<
      ADORA::ADORADialect, 
      // ADORA::ADORATensor::ADORATensorDialect, 
      KrnlDialect, affine::AffineDialect, tensor::TensorDialect,
      arith::ArithDialect, func::FuncDialect, linalg::LinalgDialect,
      math::MathDialect, vector::VectorDialect, memref::MemRefDialect,
      shape::ShapeDialect, scf::SCFDialect, cf::ControlFlowDialect>();
    
  RewritePatternSet patterns(&getContext());
  ADORATypeConverter typeConverter;
  ///////////////////////////////////////
  // 0th stage: fuse some onnx operator to adora tensor operator
  ///////////////////////////////////////
  FuseONNXOperatorToAdoraTensor(module);
  patterns.clear();

  ///////////////////////////////////////
  // 1st stage: outline specific operators
  ///////////////////////////////////////
  markElementWiseOpDynamicIllegal(target); /// For elementwize op, outline it when tensor is big
  populateONNXOutlinePattern(patterns, &getContext(), typeConverter);
  // if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
  //   signalPassFailure();
  // }

  // patterns.clear();
  populateADORATensorOutlinePattern(patterns, &getContext(), typeConverter);
  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }


  LLVM_DEBUG(llvm::errs() << "[DEBUG]after outlining:\n" ; module.dump(););
  patterns.clear();

  ///////////////////////////////////////
  // 2nd stage: onnx to krnl
  ///////////////////////////////////////
  ConversionTarget targetTokrnl(getContext());
  targetTokrnl.addLegalDialect<
      ADORA::ADORADialect, ADORA::ADORATensor::ADORATensorDialect, 
      KrnlDialect, affine::AffineDialect, tensor::TensorDialect,
      arith::ArithDialect, func::FuncDialect, linalg::LinalgDialect,
      math::MathDialect, vector::VectorDialect, memref::MemRefDialect,
      shape::ShapeDialect, scf::SCFDialect, cf::ControlFlowDialect>();

  // Convert types to legal types for the Krnl dialect.
  targetTokrnl.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
    // FuncOp is legal only if types have been converted to Std types.
    return typeConverter.isSignatureLegal(op.getFunctionType());
  });

  targetTokrnl.addDynamicallyLegalOp<func::CallOp>([&](func::CallOp op) {
    // CallOp is legal only if types have been converted to Std types.
    return typeConverter.isLegal(op);
  });

  // Operations that are legal only if types are not tensors.
  targetTokrnl.addDynamicallyLegalOp<mlir::func::ReturnOp>([&](Operation *op) {
    return llvm::none_of(op->getOperandTypes(),
        [](Type type) { return mlir::isa<TensorType>(type); });
  });

  targetTokrnl.addDynamicallyLegalOp<ADORA::ADORATensor::GemmOp>([&](auto op) {
    auto hasTensor = llvm::any_of(op->getOperandTypes(), [](Type t){ return t.isa<TensorType>(); }) ||
                     llvm::any_of(op->getResultTypes(), [](Type t){ return t.isa<TensorType>(); });
    return !hasTensor;
  });

  targetTokrnl.addDynamicallyLegalOp<ADORA::ADORATensor::MatMulOp>([&](auto op) {
    auto hasTensor = llvm::any_of(op->getOperandTypes(), [](Type t){ return t.isa<TensorType>(); }) ||
                     llvm::any_of(op->getResultTypes(), [](Type t){ return t.isa<TensorType>(); });
    return !hasTensor;
  });

  // Define patterns.
  populateONNXToKrnlConversionPatternInAdora(patterns, typeConverter, &getContext(), dimAnalysis);
  if (failed(applyPartialConversion(module, targetTokrnl, std::move(patterns)))) {
    signalPassFailure();
  }
  LLVM_DEBUG(llvm::errs() << "[DEBUG]after onnx-to-krnl:\n" ; module.dump(););
  patterns.clear();

  ///////////////////////////////////////
  // 2.5 stage: handle batched ADORATensor.Gemm
  ///////////////////////////////////////
  lowerBatchedADORAGemmOps(module);
  LLVM_DEBUG(llvm::errs() << "[DEBUG]after batched adora gemm:\n" ; module.dump(););

  ///////////////////////////////////////
  // 3rd stage: krnl to affine
  ///////////////////////////////////////
  ConversionTarget targetToaffine(getContext());
  // target.clear();
  module.walk([&](func::FuncOp funcOp) {
    const auto &dataLayoutAnalysis = getAnalysis<DataLayoutAnalysis>();
    onnx_mlir::krnl::lowerKrnlIteratesOpDefineLoopOpAndUnrollOp(funcOp, dataLayoutAnalysis);
  });
  LLVM_DEBUG(llvm::errs() << "[DEBUG]after lowerKrnlIteratesOpDefineLoopOpAndUnrollOp:\n" ; module.dump(););
  targetToaffine.addIllegalOp<KrnlTerminatorOp>();
  targetToaffine.addIllegalOp<KrnlMatMulOp>();
  targetToaffine.addIllegalOp<KrnlCopyToBufferOp>();
  targetToaffine.addIllegalOp<KrnlCopyFromBufferOp>();
  targetToaffine.addIllegalOp<KrnlPrefetchOp>();
  targetToaffine.addLegalOp<KrnlParallelClauseOp>();
  targetToaffine.addLegalOp<mlir::affine::AffineYieldOp>();
  targetToaffine.addLegalOp<mlir::affine::AffineLoadOp>();
  targetToaffine.addLegalOp<mlir::affine::AffineStoreOp>();
  targetToaffine.addLegalOp<KrnlVectorTypeCastOp>();
  targetToaffine.addLegalOp<UnrealizedConversionCastOp>();

  targetToaffine.addLegalDialect<mlir::affine::AffineDialect, 
      mlir::arith::ArithDialect,
      mlir::memref::MemRefDialect, mlir::func::FuncDialect,
      mlir::vector::VectorDialect,
      ADORA::ADORATensor::ADORATensorDialect, ADORA::ADORADialect>();

  patterns.clear();
  krnl::populateKrnlToAffineConversion(typeConverter, patterns, &getContext());
  populateAdoraLoweringKrnlGlobalToMemRefGlobal(patterns, &getContext());
  if (failed(applyPartialConversion(module, targetToaffine, std::move(patterns)))) {
    signalPassFailure();
  }

  ///////////////////////////////////////
  // 4th stage: unconverted krnl ops to affine
  ///////////////////////////////////////
  // targetToaffine.addIllegalOp<KrnlMemcpyOp>();
  // patterns.clear();
  // if (failed(applyPartialConversion(module, targetToaffine,
  //                                   std::move(patterns))))
  //   signalPassFailure();
}
// func::FuncOp LinalgToSystolicGEMMPass::ConvertMatmulToSystolic(linalg::MatmulOp matmul){
//   llvm::SetVector<mlir::Value> operands;
//   for(auto operand : matmul.getOperands()){
//     operands.insert(operand);
//   }
//   func::FuncOp func = ConvertMatmulToFunc(matmul, operands, "matmul_" + std::to_string(matmul_idx));
  
//   //// Annotate the function as a systolic gemm
  

//   //// lower matmul to gemm
//   linalg::MatmulOp ToLowerOp = dyn_cast<linalg::MatmulOp>(func.getBody().front().front());
//   ADORATensor::GemmOp newGemm;
//   newGemm = ConvertToSameADORATensorOp<ADORATensor::GemmOp>(ToLowerOp);

//   //// set to a 4x4 weight stationary
//   ADORATensor::SystolicImplInterface Sinterface(newGemm);
//   // mlir::SmallVector<int64_t> tile = {4,4};
//   Sinterface.setStationaryKind(MatMulStrategy::WeightStationary);
//   Sinterface.setTileSize(ArrayRef<int64_t>({4,4}));

//   matmul_idx++;
//   return func;
// }


std::unique_ptr<OperationPass<ModuleOp>> createConvertONNXLayersToAdoraPass() {
  return std::make_unique<ConvertONNXLayersToAdoraPass>();
}

}
} // namespace
} // namespace

// std::unique_ptr<OperationPass<ModuleOp>> ::mlir::ADORA::createLinalgToSystolicGEMMPass() {
//   return std::make_unique<LinalgToSystolicGEMMPass>();
// }
