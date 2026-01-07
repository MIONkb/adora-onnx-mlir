//===- SetBatchSize.cpp - Set batch size for ONNX entry tensors -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include "ADORAONNX/Transform/Passes.h"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXDimAnalysis.hpp"

#define DEBUG_TYPE "adora-onnx-set-batch"

namespace mlir {
namespace ADORA {
namespace ADORATensor {

namespace {
struct SetONNXBatchSizePass
    : public SetONNXBatchSizePassBase<SetONNXBatchSizePass> {
  void runOnOperation() final;
};

Type setBatchDim(Type type, int64_t batchSize) {
  auto rankedType = type.dyn_cast<RankedTensorType>();
  if (!rankedType || rankedType.getRank() == 0)
    return type;

  SmallVector<int64_t, 4> shape(rankedType.getShape().begin(),
      rankedType.getShape().end());
  if (shape[0] == batchSize)
    return type;
  shape[0] = batchSize;
  return RankedTensorType::get(
      shape, rankedType.getElementType(), rankedType.getEncoding());
}

bool hasBatchDimParam(ArrayAttr attrs, unsigned index) {
  if (!attrs || index >= attrs.size())
    return false;
  auto dictAttr = llvm::dyn_cast<DictionaryAttr>(attrs[index]);
  if (!dictAttr || !dictAttr.contains("onnx.dim_params"))
    return false;
  StringRef dimParams = mlir::cast<StringAttr>(
      dictAttr.getNamed("onnx.dim_params").value().getValue())
                            .getValue();
  SmallVector<StringRef, 4> splittedDimParams;
  dimParams.split(splittedDimParams, ',');
  for (StringRef entry : splittedDimParams) {
    auto indexParam = entry.split(':');
    if (indexParam.first == "0") {
      std::string lowered = indexParam.second.lower();
      if (StringRef(lowered).contains("batch"))
        return true;
    }
  }
  return false;
}

bool updateFunctionSignature(func::FuncOp func, int64_t batchSize,
    ArrayRef<bool> updateInputs, ArrayRef<bool> updateResults) {
  auto functionType = func.getFunctionType();
  SmallVector<Type, 4> newInputs;
  SmallVector<Type, 4> newResults;
  newInputs.reserve(functionType.getNumInputs());
  newResults.reserve(functionType.getNumResults());

  bool changed = false;
  for (auto [type, shouldUpdate] :
      llvm::zip(functionType.getInputs(), updateInputs)) {
    Type newType = shouldUpdate ? setBatchDim(type, batchSize) : type;
    if (newType != type)
      changed = true;
    newInputs.push_back(newType);
  }
  for (auto [type, shouldUpdate] :
      llvm::zip(functionType.getResults(), updateResults)) {
    Type newType = shouldUpdate ? setBatchDim(type, batchSize) : type;
    if (newType != type)
      changed = true;
    newResults.push_back(newType);
  }

  if (!changed)
    return false;

  if (!func.isExternal()) {
    Block &entry = func.getBody().front();
    for (auto [arg, newType] : llvm::zip(entry.getArguments(), newInputs))
      arg.setType(newType);
  }

  auto newFunctionType =
      FunctionType::get(func.getContext(), newInputs, newResults);
  func.setType(newFunctionType);
  return true;
}
} // namespace

void SetONNXBatchSizePass::runOnOperation() {
  ModuleOp module = getOperation();

  if (batchSize <= 0) {
    module.emitError() << "batch size must be > 0, got " << batchSize;
    signalPassFailure();
    return;
  }

  onnx_mlir::DimAnalysis dimAnalysis(module);
  dimAnalysis.analyze();

  bool updated = false;
  bool hasEntryPoint = false;
  module.walk([&](ONNXEntryPointOp entryOp) {
    auto funcAttr = entryOp->getAttrOfType<SymbolRefAttr>(
        ONNXEntryPointOp::getEntryPointFuncAttrName());
    if (!funcAttr)
      return;
    auto *entryPointOp = module.lookupSymbol(funcAttr.getLeafReference());
    auto entryFunc = dyn_cast_or_null<func::FuncOp>(entryPointOp);
    if (!entryFunc)
      return;

    hasEntryPoint = true;
    auto uses = entryFunc.getSymbolUses(module);
    if (uses) {
      for (const SymbolTable::SymbolUse &use : *uses) {
        if (isa<func::CallOp>(use.getUser()))
          return;
      }
    }

    SmallVector<onnx_mlir::DimAnalysis::DimT, 4> batchDims;
    ArrayAttr argAttrs = entryFunc.getArgAttrsAttr();
    for (auto [index, arg] : llvm::enumerate(entryFunc.getArguments())) {
      auto rankedType = arg.getType().dyn_cast<RankedTensorType>();
      if (!rankedType || rankedType.getRank() == 0)
        continue;
      if (hasBatchDimParam(argAttrs, index))
        batchDims.emplace_back(arg, 0);
    }

    if (batchDims.empty()) {
      entryFunc.emitWarning()
          << "no onnx.dim_params batch annotation found on entry arguments";
      return;
    }

    auto isBatchDim = [&](Value value) {
      auto rankedType = value.getType().dyn_cast<RankedTensorType>();
      if (!rankedType || rankedType.getRank() == 0)
        return false;
      for (const auto &batchDim : batchDims) {
        if (batchDim.first == value && batchDim.second == 0)
          return true;
        if (dimAnalysis.sameDim(value, 0, batchDim.first, batchDim.second))
          return true;
      }
      return false;
    };

    SmallVector<bool, 4> updateInputs;
    updateInputs.reserve(entryFunc.getNumArguments());
    for (BlockArgument arg : entryFunc.getArguments())
      updateInputs.push_back(isBatchDim(arg));

    SmallVector<bool, 4> updateResults;
    if (!entryFunc.isExternal()) {
      Operation *terminator = entryFunc.getBody().back().getTerminator();
      if (auto returnOp = dyn_cast<func::ReturnOp>(terminator)) {
        for (Value result : returnOp.getOperands())
          updateResults.push_back(isBatchDim(result));
      } else if (auto returnOp = dyn_cast<ONNXReturnOp>(terminator)) {
        for (Value result : returnOp.getOperands())
          updateResults.push_back(isBatchDim(result));
      }
    }
    updateResults.resize(entryFunc.getNumResults(), false);

    if (updateFunctionSignature(entryFunc, batchSize, updateInputs,
            updateResults))
      updated = true;

    entryFunc.walk([&](Operation *op) {
      for (Value result : op->getResults()) {
        if (!isBatchDim(result))
          continue;
        Type newType = setBatchDim(result.getType(), batchSize);
        if (newType != result.getType()) {
          result.setType(newType);
          updated = true;
        }
      }
    });
    for (Block &block : entryFunc.getBody()) {
      for (BlockArgument arg : block.getArguments()) {
        if (!isBatchDim(arg))
          continue;
        Type newType = setBatchDim(arg.getType(), batchSize);
        if (newType != arg.getType()) {
          arg.setType(newType);
          updated = true;
        }
      }
    }
  });

  if (!hasEntryPoint) {
    module.walk([&](func::FuncOp func) {
      if (func.isExternal())
        return;
      auto uses = func.getSymbolUses(module);
      if (uses) {
        for (const SymbolTable::SymbolUse &use : *uses) {
          if (isa<func::CallOp>(use.getUser()))
            return;
        }
      }

      SmallVector<onnx_mlir::DimAnalysis::DimT, 4> batchDims;
      ArrayAttr argAttrs = func.getArgAttrsAttr();
      for (auto [index, arg] : llvm::enumerate(func.getArguments())) {
        auto rankedType = arg.getType().dyn_cast<RankedTensorType>();
        if (!rankedType || rankedType.getRank() == 0)
          continue;
        if (hasBatchDimParam(argAttrs, index))
          batchDims.emplace_back(arg, 0);
      }
      if (batchDims.empty())
        return;

      auto isBatchDim = [&](Value value) {
        auto rankedType = value.getType().dyn_cast<RankedTensorType>();
        if (!rankedType || rankedType.getRank() == 0)
          return false;
        for (const auto &batchDim : batchDims) {
          if (batchDim.first == value && batchDim.second == 0)
            return true;
          if (dimAnalysis.sameDim(value, 0, batchDim.first, batchDim.second))
            return true;
        }
        return false;
      };

      SmallVector<bool, 4> updateInputs;
      updateInputs.reserve(func.getNumArguments());
      for (BlockArgument arg : func.getArguments())
        updateInputs.push_back(isBatchDim(arg));

      SmallVector<bool, 4> updateResults;
      Operation *terminator = func.getBody().back().getTerminator();
      if (auto returnOp = dyn_cast<func::ReturnOp>(terminator)) {
        for (Value result : returnOp.getOperands())
          updateResults.push_back(isBatchDim(result));
      } else if (auto returnOp = dyn_cast<ONNXReturnOp>(terminator)) {
        for (Value result : returnOp.getOperands())
          updateResults.push_back(isBatchDim(result));
      }
      updateResults.resize(func.getNumResults(), false);

      if (updateFunctionSignature(func, batchSize, updateInputs, updateResults))
        updated = true;

      func.walk([&](Operation *op) {
        for (Value result : op->getResults()) {
          if (!isBatchDim(result))
            continue;
          Type newType = setBatchDim(result.getType(), batchSize);
          if (newType != result.getType()) {
            result.setType(newType);
            updated = true;
          }
        }
      });
      for (Block &block : func.getBody()) {
        for (BlockArgument arg : block.getArguments()) {
          if (!isBatchDim(arg))
            continue;
          Type newType = setBatchDim(arg.getType(), batchSize);
          if (newType != arg.getType()) {
            arg.setType(newType);
            updated = true;
          }
        }
      }
    });
  }

  if (!updated) {
    module.emitWarning() << "no eligible functions updated for batch size "
                         << batchSize;
  }
}

std::unique_ptr<OperationPass<ModuleOp>> createSetONNXBatchSizePass() {
  return std::make_unique<SetONNXBatchSizePass>();
}

} // namespace ADORATensor
} // namespace ADORA
} // namespace mlir
