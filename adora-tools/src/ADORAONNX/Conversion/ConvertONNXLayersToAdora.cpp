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

template <typename T> inline void setADORAKernelFunctionAttr(T op){
  op.getOperation()->setAttr("adora_kernel", mlir::UnitAttr::get(op.getContext()));
}

static bool hasDynamicShape(mlir::Type ty) {
  if (auto st = ty.dyn_cast<mlir::ShapedType>())
    return !st.hasStaticShape();
  return false; 
}

template <typename RangeT>
static bool anyDynamic(const RangeT &tys) {
  for (mlir::Type t : tys)
    if (hasDynamicShape(t)) return true;
  return false;
}

/// @brief ToFunc
/// @param Op
/// @param FnName
/// @param operands
/// @return
template <typename OpT>
static std::pair<func::CallOp, func::FuncOp> ConvertONNXOpTtoFunc(
  OpT op, llvm::SmallVector<mlir::Type, 8> resultTypes, mlir::ValueRange &operands, std::string FnName)
{
  Location loc = op.getLoc();
  // Create a builder with no insertion point, insertion will happen separately
  // due to symbol table manipulation
  mlir::MLIRContext *ctx = op.getContext();
  OpBuilder builder(ctx);
  mlir::ModuleOp module = op.getOperation()->template getParentOfType<mlir::ModuleOp>();
  builder.setInsertionPointToEnd(module.getBody());

  /////////////////////////
  /// get input types and output types
  /////////////////////////
  llvm::SmallVector<mlir::Type, 8> funcinputTypes;
  funcinputTypes.reserve(operands.size());
  for (mlir::Value v : operands) {
    funcinputTypes.push_back(v.getType());
  }

  // for (mlir::Value v : allocs)   {
  //   funcinputTypes.push_back(v.getType());
  // }


  auto fnType = mlir::FunctionType::get(ctx, funcinputTypes, /*results*/ resultTypes);
  auto Func = builder.create<func::FuncOp>(loc, FnName, fnType);

  // KernelFunc->setAttr(kernelFnName, builder.getUnitAttr());
  // KernelFunc->setAttr("Kernel", builder.getUnitAttr());

  /// Pass func arguements outside of KernelOp
  Block *entryBlock = Func.addEntryBlock();
  func::FuncOp::BlockArgListType args = entryBlock->getArguments();;
  builder.setInsertionPointToStart(entryBlock);

  mlir::IRMapping mapping;
  // // Block &entryBlock = KernelFunc.getBody().front();
  for (unsigned index = 0; index < operands.size(); index++)
  {
    mapping.map(op.getOperation()->getOperand(index), entryBlock->getArgument(index));
  }
  mlir::Operation* newop = op.getOperation()->clone(mapping);
  entryBlock->push_back(newop);

  llvm::SmallVector<mlir::Value> returnValues;
  for (uint64_t i = 0; i < newop->getResults().size(); i++) {
    if(!newop->getResult(i).getType().isa<mlir::NoneType>()){
      returnValues.push_back(newop->getResult(i));
    }
  }
  // entryBlock->push_back(builder.create<func::ReturnOp>(newop->getLoc(), dyn_cast<OpT>(newop).getResultTensors()));
  builder.create<func::ReturnOp>(newop->getLoc(), returnValues);
  // builder.create<func::ReturnOp>(newop->getLoc(), newop->getResults());
  //// specific it as a kernel
  // ::mlir::ADORA::specifyOneOperationToADORAKernel(newop, FnName);
  LLVM_DEBUG(llvm::errs() << "[debug] after create:\n"; Func.dump(););

  builder.setInsertionPoint(op);
  llvm::SmallVector<mlir::Value, 8> callArgs;
  callArgs.append(operands.begin(), operands.end());
  // callArgs.append(allocs.begin(), allocs.end());

  func::CallOp callop = builder.create<func::CallOp>(loc, Func, callArgs);

  setStringAttr(Func, "onnx_layer", op->getName().getStringRef().str());
  setADORAKernelFunctionAttr(Func);
  if(op.getOperation()->hasAttr("onnx_node_name"))
    callop.getOperation() -> setAttr("onnx_node_name", op.getOperation()->getAttr("onnx_node_name"));

  return std::make_pair(callop, Func);
}

#define kTensorSizeThreshold 32

bool isLargeTensor(mlir::Value tensor) {
  auto type = tensor.getType();

  if (auto shaped = type.dyn_cast<mlir::ShapedType>()) {
    if (!shaped.hasStaticShape())
      return true; 

    int64_t elems = 1;
    for (auto dim : shaped.getShape())
      elems *= dim;

    return elems >= kTensorSizeThreshold;
  }

  return false;
}

void markElementWiseOpDynamicIllegal(ConversionTarget& target){
  target.addDynamicallyLegalOp<ONNXMulOp>([&](ONNXMulOp op) {
    return !isLargeTensor(op.getA()) && !isLargeTensor(op.getB());
  });
  target.addDynamicallyLegalOp<ONNXAddOp>([&](ONNXAddOp op) {
    return !isLargeTensor(op.getA()) && !isLargeTensor(op.getB());
  });
}

// Template to create ONNXOp to Call pattern.
template <typename OP_TYPE, typename SHAPEHELPER_TYPE>
struct ONNXGenericOpToFuncCall : public mlir::OpConversionPattern<OP_TYPE> {
  using ADAPTOR_TYPE = typename OP_TYPE::Adaptor;
  ONNXGenericOpToFuncCall(
    // mlir::TypeConverter &typeConverter,
      mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<OP_TYPE>(
            /*typeConverter,*/ ctx, /*benefit higher than default*/ 10){}

  LogicalResult matchAndRewrite(OP_TYPE onnxOp, typename OP_TYPE::Adaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override{
    mlir::Operation *op = onnxOp.getOperation();
    mlir::Location loc = op->getLoc();
    mlir::ValueRange operands = adaptor.getOperands();
    mlir::ValueRange results = op->getResults();
    
    mlir::ModuleOp module = op->template getParentOfType<mlir::ModuleOp>();
    mlir::SymbolTable symTab(module);

    // Get shape.
    // MultiDialectBuilder<IndexExprBuilderForKrnl, MemRefBuilder> create(
    //     rewriter, loc);

    // SHAPEHELPER_TYPE shapeHelper(op, operands, &create.krnlIE);
    // onnx_mlir::ONNXOpShapeHelper *getShapeHelper(mlir::Operation *op, llvm::ArrayRef<mlir::Value> operands, onnx_mlir::IndexExprBuilder *ieBuilder, onnx_mlir::IndexExprScope *scope);
    // SHAPEHELPER_TYPE shapeHelper(op, operands, &create.krnlIE, nullptr);
    // shapeHelper.computeShapeAndAssertOnFailure();
    // Insert an allocation and deallocation for the result of this operation.
    // std::vector<mlir::Value> allocs = allocForONNXOp<OP_TYPE>(
    //     onnxOp, rewriter, this->typeConverter, shapeHelper);

    // Create func.call and func.func here.
    // Check whether to assign a new func or use existing func
    llvm::SmallVector<mlir::Type, 8> funcInputTypes, resultTypes;
    funcInputTypes.reserve(operands.size());
    for (mlir::Value v : operands) {
      funcInputTypes.push_back(v.getType());
    }
    // for (mlir::Value v : allocs) {
    //   funcInputTypes.push_back(v.getType());
    // }

    resultTypes.reserve(results.size());
    for (uint64_t i = 0; i < results.size(); i++) {
      if(!results[i].getType().isa<mlir::NoneType>()){
        resultTypes.push_back(results[i].getType());
      }
    }

    const bool forceNewForDynamic = anyDynamic(funcInputTypes) || anyDynamic(resultTypes);
    std::string opName = op->getName().getStringRef().str().substr(5);
    func::CallOp callop;
    if (forceNewForDynamic) {
      unsigned n = 0;
      std::string funcName = opName + "_" + std::to_string(0);
      for (auto func : module.getOps<func::FuncOp>()) {
        auto attr = func->getAttrOfType<StringAttr>("onnx_layer");
        if (attr && attr.getValue() == opName){
          funcName = func.getSymName().str() + "_" + std::to_string(++n);
        }
      }

      auto funcpair = ConvertONNXOpTtoFunc(mlir::dyn_cast<OP_TYPE>(op), resultTypes, operands, funcName);
      callop = funcpair.first;
      func::FuncOp funcop = funcpair.second;
      symTab.insert(funcop);
    } else {
      // all tensors are static tensor
      unsigned n = 0;
      std::string funcName = opName + "_" + std::to_string(0);
      for (auto func : module.getOps<func::FuncOp>()) {
        auto attr = func->getAttrOfType<StringAttr>("onnx_layer");
        if (attr && attr.getValue() == opName){
          FunctionType funcType = func.getFunctionType();
          if (funcType.getInputs() == ArrayRef(funcInputTypes)) {
            funcName = func.getSymName().str();
            break;
          } 
          else{
            funcName = opName + "_" + std::to_string(++n);
          }
        }
      }

      /// find no reusable functions, create a new one
      mlir::func::FuncOp callee = symTab.lookup<mlir::func::FuncOp>(funcName);
      if (!callee) {
        auto funcpair = ConvertONNXOpTtoFunc(mlir::dyn_cast<OP_TYPE>(op), resultTypes, operands, funcName);
        callop = funcpair.first;
        func::FuncOp funcop = funcpair.second;
        setStringAttr(funcop, "onnx_layer", opName);
        setADORAKernelFunctionAttr(funcop);
        symTab.insert(funcop);
      }
      else{
        llvm::SmallVector<mlir::Value, 8> callArgs;
        callArgs.append(operands.begin(), operands.end());
        // callArgs.append(allocs.begin(), allocs.end());
        callop = rewriter.create<func::CallOp>(loc, callee, callArgs);
        if(op->hasAttr("onnx_node_name"))
          callop.getOperation() -> setAttr("onnx_node_name", op->getAttr("onnx_node_name"));
      }
    }

    // func::FuncOp newfunc = ConvertONNXOpTtoFunc(dyn_cast<OP_TYPE>(op), resultTypes, operands, funcName);  
    // std::vector<mlir::Value> allocs_convert;
    // for(int index = 0; index < allocs.size(); index++){
    //   allocs_convert.push_back(
    //     rewriter.create<UnrealizedConversionCastOp>(loc, resultTypes[index], allocs[index]).getResult(0));
    // }
    // LLVM_DEBUG(llvm::errs() << "before replace:"; module.dump(););
    if(results.size() != callop.getResults().size()){
      int index = 0;
      for (uint64_t i = 0; i < results.size(); i++) {
        if(!results[i].getType().isa<mlir::NoneType>()){
          rewriter.replaceAllUsesWith(results[i], callop.getResult(index++));
        }
      }
      op->erase();
    }
    else{
      rewriter.replaceOp(op, callop);
    }

    // LLVM_DEBUG(llvm::errs() << "after replace:"; module.dump();); 
    return success();
  }
};

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

void populateONNXOutlinePattern(RewritePatternSet &patterns, 
  MLIRContext *ctx, ADORATypeConverter typeConverter) 
{ 
  patterns.insert<
    ONNXGenericOpToFuncCall<ONNXMatMulOp, onnx_mlir::ONNXMatMulOpShapeHelper>,
    ONNXGenericOpToFuncCall<ONNXAddOp, onnx_mlir::ONNXAddOpShapeHelper>,
    ONNXGenericOpToFuncCall<ONNXMulOp, onnx_mlir::ONNXMulOpShapeHelper>,
    ONNXGenericOpToFuncCall<ONNXRMSLayerNormalizationOp, onnx_mlir::ONNXRMSLayerNormalizationOpShapeHelper>,
    ONNXGenericOpToFuncCall<ONNXLayerNormalizationOp, onnx_mlir::ONNXLNOpShapeHelper<mlir::ONNXLayerNormalizationOp>>
  >(ctx);
}

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
  if (failed(applyPartialConversion(module, targetToaffine, std::move(patterns)))) {
    signalPassFailure();
  }

  ///////////////////////////////////////
  // 4th stage: unconverted krnl ops to affine
  ///////////////////////////////////////
  // targetToaffine.addIllegalOp<KrnlMemcpyOp>();
  // patterns.clear();
  // // 然后 applyConversion
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
