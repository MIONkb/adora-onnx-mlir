//===- LinalgToSystolicGEMM.cpp - conversion from Linalg named operators to systolic gemm --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// #include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
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
      ADORA::ADORADialect, ADORA::ADORATensor::ADORATensorDialect, 
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

  // Define patterns.
  populateONNXToKrnlConversionPatternInAdora(patterns, typeConverter, &getContext(), dimAnalysis);
  if (failed(applyPartialConversion(module, targetTokrnl, std::move(patterns)))) {
    signalPassFailure();
  }
  LLVM_DEBUG(llvm::errs() << "[DEBUG]after onnx-to-krnl:\n" ; module.dump(););
  patterns.clear();

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
