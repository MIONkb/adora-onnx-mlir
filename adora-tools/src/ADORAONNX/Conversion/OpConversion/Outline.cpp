#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
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
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
// #include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
// #include "src/Dialect/Mlir/VectorMachineSupport.hpp"
// #include "PassDetail.h"

#define DEBUG_TYPE "onnx-to-adora"

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

// Template to create ONNXOp to Call pattern.
// template <typename OP_TYPE, typename SHAPEHELPER_TYPE>
template <typename OP_TYPE>
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


//===----------------------------------------------------------------------===//
// ADORATensor outline: Op -> func.call + (new) func.func
//===----------------------------------------------------------------------===//

static std::string getOpShortName(Operation *op) {
  // e.g. "onnx.MatMul" -> "MatMul"
  //      "ADORATensor.Gemm" -> "Gemm"
  StringRef full = op->getName().getStringRef();
  size_t dot = full.rfind('.');
  if (dot == StringRef::npos) return full.str();
  return full.substr(dot + 1).str();
}

/// Outline the given op into a new callee, and replace the op with func.call.
/// This is ADORA-flavored outline:
/// - marks the created function with `adora_kernel`
/// - optionally propagates `onnx_node_name` to the call site
/// - does NOT set "onnx_layer" on the created function (to avoid semantic abuse)
template <typename OpT>
static std::pair<func::CallOp, func::FuncOp>
ConvertADORATensorOpTtoFunc(OpT op,
                            SmallVector<Type, 8> resultTypes,
                            ValueRange operands,
                            StringRef fnName) {
  Location loc = op.getLoc();
  MLIRContext *ctx = op.getContext();
  ModuleOp module = op->getParentOfType<ModuleOp>();

  // Build the callee in module scope.
  OpBuilder moduleBuilder(ctx);
  moduleBuilder.setInsertionPointToEnd(module.getBody());

  SmallVector<Type, 8> inputTypes;
  inputTypes.reserve(operands.size());
  for (Value v : operands)
    inputTypes.push_back(v.getType());

  auto fnType = FunctionType::get(ctx, inputTypes, resultTypes);
  auto funcOp = moduleBuilder.create<func::FuncOp>(loc, fnName, fnType);

  // Mark as kernel function for downstream extraction passes.
  setADORAKernelFunctionAttr(funcOp);

  // Populate function body: clone the op and return its results.
  Block *entry = funcOp.addEntryBlock();
  moduleBuilder.setInsertionPointToStart(entry);

  IRMapping mapping;
  for (unsigned i = 0; i < operands.size(); ++i)
    mapping.map(op.getOperation()->getOperand(i), entry->getArgument(i));

  Operation *cloned = op.getOperation()->clone(mapping);
  entry->push_back(cloned);

  SmallVector<Value, 8> rets;
  rets.reserve(cloned->getNumResults());
  for (Value r : cloned->getResults())
    if (!r.getType().isa<NoneType>())
      rets.push_back(r);

  moduleBuilder.create<func::ReturnOp>(loc, rets);

  // Replace original op with a call.
  OpBuilder callBuilder(op);
  SmallVector<Value, 8> callArgs(operands.begin(), operands.end());
  auto callOp = callBuilder.create<func::CallOp>(loc, funcOp, callArgs);

  // Propagate node name if present.
  if (op->hasAttr("onnx_node_name"))
    callOp->setAttr("onnx_node_name", op->getAttr("onnx_node_name"));

  return {callOp, funcOp};
}

template <typename OP_TYPE>
struct ADORATensorGenericOpToFuncCall : public OpConversionPattern<OP_TYPE> {
  using OpConversionPattern<OP_TYPE>::OpConversionPattern;

  LogicalResult matchAndRewrite(OP_TYPE op, typename OP_TYPE::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Operation *rawOp = op.getOperation();
    Location loc = rawOp->getLoc();
    ModuleOp module = rawOp->getParentOfType<ModuleOp>();
    SymbolTable symTab(module);

    ValueRange operands = adaptor.getOperands();
    ValueRange results = rawOp->getResults();

    // Collect signature types.
    SmallVector<Type, 8> inputTypes, resultTypes;
    inputTypes.reserve(operands.size());
    for (Value v : operands)
      inputTypes.push_back(v.getType());

    resultTypes.reserve(results.size());
    for (Value r : results)
      if (!r.getType().isa<NoneType>())
        resultTypes.push_back(r.getType());

    const bool forceNewForDynamic = anyDynamic(inputTypes) || anyDynamic(resultTypes);

    // Use a stable base name derived from opname.
    // e.g. "ADORATensor.Gemm" -> "Gemm_0"
    std::string opShort = getOpShortName(rawOp);
    std::string funcName = opShort + "_0";

    func::CallOp callOp;

    auto findReusable = [&](StringRef baseLayerName) -> func::FuncOp {
      // Reuse only by exact signature match.
      for (auto f : module.getOps<func::FuncOp>()) {
        // Reuse only ADORA-outlined funcs (marked by adora_kernel).
        if (!f->hasAttrOfType<UnitAttr>("adora_kernel"))
          continue;

        if (!f.getSymName().startswith(baseLayerName))
          continue;

        if (f.getFunctionType().getInputs() == ArrayRef(inputTypes) &&
            f.getFunctionType().getResults() == ArrayRef(resultTypes))
          return f;
      }
      return nullptr;
    };

    if (!forceNewForDynamic) {
      if (auto reusable = findReusable(opShort)) {
        rewriter.setInsertionPoint(rawOp);
        SmallVector<Value, 8> callArgs(operands.begin(), operands.end());
        callOp = rewriter.create<func::CallOp>(loc, reusable, callArgs);
        if (rawOp->hasAttr("onnx_node_name"))
          callOp->setAttr("onnx_node_name", rawOp->getAttr("onnx_node_name"));
      } else {
        // Create a new function with a unique suffix.
        unsigned n = 0;
        std::string base = opShort;
        do {
          funcName = base + "_" + std::to_string(n++);
        } while (symTab.lookup(funcName));

        auto pair = ConvertADORATensorOpTtoFunc(op, resultTypes, operands, funcName);
        callOp = pair.first;
        symTab.insert(pair.second);
      }
    } else {
      // For dynamic shapes, always create a new unique function.
      unsigned n = 0;
      std::string base = opShort;
      do {
        funcName = base + "_" + std::to_string(n++);
      } while (symTab.lookup(funcName));

      auto pair = ConvertADORATensorOpTtoFunc(op, resultTypes, operands, funcName);
      callOp = pair.first;
      symTab.insert(pair.second);
    }

    // Replace uses.
    if (results.size() != callOp.getResults().size()) {
      int idx = 0;
      for (Value r : results) {
        if (!r.getType().isa<NoneType>())
          rewriter.replaceAllUsesWith(r, callOp.getResult(idx++));
      }
      rawOp->erase();
    } else {
      rewriter.replaceOp(rawOp, callOp.getResults());
    }

    return success();
  }
};

void populateONNXOutlinePattern(RewritePatternSet &patterns, 
  MLIRContext *ctx, ADORATypeConverter typeConverter) 
{ 
  patterns.insert<
    // ONNXGenericOpToFuncCall<ONNXMatMulOp, onnx_mlir::ONNXMatMulOpShapeHelper>,
    // ONNXGenericOpToFuncCall<ONNXAddOp, onnx_mlir::ONNXAddOpShapeHelper>,
    // ONNXGenericOpToFuncCall<ONNXMulOp, onnx_mlir::ONNXMulOpShapeHelper>,
    // ONNXGenericOpToFuncCall<ONNXRMSLayerNormalizationOp, onnx_mlir::ONNXRMSLayerNormalizationOpShapeHelper>,
    // ONNXGenericOpToFuncCall<ONNXLayerNormalizationOp, onnx_mlir::ONNXLNOpShapeHelper<mlir::ONNXLayerNormalizationOp>>

    ONNXGenericOpToFuncCall<ONNXMatMulOp>,
    ONNXGenericOpToFuncCall<ONNXAddOp>,
    ONNXGenericOpToFuncCall<ONNXMulOp>,
    ONNXGenericOpToFuncCall<ONNXRMSLayerNormalizationOp>,
    ONNXGenericOpToFuncCall<ONNXLayerNormalizationOp>
  >(ctx);

  patterns.insert<
      ADORATensorGenericOpToFuncCall<mlir::ADORA::ADORATensor::GemmOp>,
      ADORATensorGenericOpToFuncCall<mlir::ADORA::ADORATensor::MatMulOp>
  >(ctx);
}

}
} // namespace
} // namespace