//====------ ONNXToADORACommon.hpp - ONNX dialects to Krnl lowering --------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file contains common code shared by the functions performing the
// lowering to the ADORA dialect.
//
//===----------------------------------------------------------------------===//

#ifndef ADORA_ONNX_MLIR_ONNX_TO_ADORA_H
#define ADORA_ONNX_MLIR_ONNX_TO_ADORA_H

#include <map>

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/TypeSwitch.h"

// #include "src/Compiler/OptionUtils.hpp"
#include "src/Dialect/Krnl/DialectBuilder.hpp"
#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Dialect/Mlir/DialectBuilder.hpp"
#include "src/Dialect/Mlir/IndexExpr.hpp"
#include "src/Dialect/Mlir/VectorMachineSupport.hpp"
#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXDimAnalysis.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
// #include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Pass/Passes.hpp"
#include "src/Support/KrnlSupport.hpp"

//===----------------------------------------------------------------------===//
// Extends OnnxBuilder with member functions that might generate Krnl dialect
// operations.
//===----------------------------------------------------------------------===//

using namespace onnx_mlir;
namespace mlir {
namespace ADORA {
namespace ADORATensor {
//===----------------------------------------------------------------------===//
// Type conversion from Onnx types to Krnl types:
//   - from Tensor type to the Standard dialect MemRef type
//   - from onnx.StringType to krnl.StringType
//===----------------------------------------------------------------------===//

class ADORATypeConverter : public mlir::TypeConverter {
public:
  ADORATypeConverter();

  /// Return true if the inputs and outputs of the given function type are
  /// legal. [Taken from MLIR and adapted to only check the legality of the
  /// inputs. Once unranked results can be handled gracefully this
  /// override needs to be removed in favour of the original MLIR one.]
  bool isSignatureLegal(mlir::FunctionType funcType) {
    return llvm::all_of(llvm::concat<const mlir::Type>(
                            funcType.getInputs(), funcType.getResults()),
        [this](mlir::Type type) { return isLegal(type); });
  }

  /// Return true if the operands/results of call have a legal type.
  bool isSignatureLegal(mlir::func::CallOp call) {
    auto f = [this](mlir::Type type) { return isLegal(type); };
    return llvm::all_of(call.getOperandTypes(), f) &&
           llvm::all_of(call.getResultTypes(), f);
  }

  // Return the default alignment value used when allocating a MemRef buffer
  // for the given type. E.g. some special types for accelerators requires
  // 4K-aligned buffers.
  static int64_t getDefaultAllocAlignment(mlir::Type type);
};

}
} // namespace
} // namespace
#endif
