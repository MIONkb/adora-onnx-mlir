//====----- ONNXToADORACommon.cpp - ONNX dialects to Krnl lowering ---------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file contains common code shared by the functions performing the
// lowering to the KRNL dialect.
//
//===----------------------------------------------------------------------===//

// #include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"

// #include "src/Accelerators/Accelerator.hpp"
// #include "src/Compiler/CompilerOptions.hpp"
#include "src/Dialect/Krnl/DialectBuilder.hpp"
#include "src/Dialect/Mlir/DialectBuilder.hpp"
// #include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Dialect/ONNX/OnnxElementsAttrBuilder.hpp"
#include "ADORAONNX/Conversion/ONNXToADORACommon.hpp"

#define DEBUG_TYPE "onnx-to-adora"

using namespace mlir;
using namespace onnx_mlir;

namespace mlir {
namespace ADORA {
namespace ADORATensor {
//===----------------------------------------------------------------------===//
// Type conversion from Onnx types to Krnl types.
//===----------------------------------------------------------------------===//

ADORATypeConverter::ADORATypeConverter() {
  // The order of type conversion is important: later ones are tried earlier.
  addConversion([](Type type) { return type; });

  addConversion([](ONNXStringType stringType) {
    return krnl::StringType::get(stringType.getContext());
  });

  addConversion([](NoneType type) {
    return type;
  });

  addConversion([](TensorType tensorType) {
    assert(tensorType.hasRank() && "expected only ranked shapes");
    // if (mlir::isa<ONNXStringType>(tensorType.getElementType())) {
    //   Type elementType = krnl::StringType::get(tensorType.getContext());
    //   return MemRefType::get(tensorType.getShape(), elementType);
    // }
    // // Accelerators may have special versions of TensorType. Call the
    // // conversions of accelerators.
    // for (auto *accel : onnx_mlir::accel::Accelerator::getAccelerators()) {
    //   MemRefType memRefType = accel->convertTensorTypeToMemRefType(tensorType);
    //   if (memRefType)
    //     return memRefType;
    // }
    // if (hasCustomONNXTensorDataLayout(tensorType))
    //   return convertTypeWithCustomONNXDataLayoutToMemRef(tensorType);
    return MemRefType::get(tensorType.getShape(), tensorType.getElementType());
  });

  addConversion([](SeqType seqType) {
    auto seqElementType = mlir::cast<ShapedType>(seqType.getElementType());
    Type elementType = seqElementType.getElementType();
    Type seqElementConvertedType;
    if (seqElementType.hasRank()) {
      seqElementConvertedType =
          MemRefType::get(seqElementType.getShape(), elementType);
    } else {
      seqElementConvertedType = UnrankedMemRefType::get(elementType, 0);
    }
    SmallVector<int64_t, 1> dims;
    dims.emplace_back(seqType.getLength());
    llvm::ArrayRef<int64_t> shape(dims.data(), dims.size());
    return MemRefType::get(shape, seqElementConvertedType);
  });

  addSourceMaterialization([&](OpBuilder &builder, Type resultType,
                               ValueRange inputs, Location loc) -> Value {
    if (inputs.size() != 1)
      return Value();

    return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
        .getResult(0);
  });

  addTargetMaterialization([&](OpBuilder &builder, Type resultType,
                               ValueRange inputs, Location loc) -> Value {
    if (inputs.size() != 1)
      return Value();

    return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
        .getResult(0);
  });
}

int64_t ADORATypeConverter::getDefaultAllocAlignment(Type type) {
  int64_t alignment = -1;
  if (auto tensorType = mlir::dyn_cast<TensorType>(type)) {
    // Accelerators may have special versions of TensorType. Call the
    // conversions of accelerators.
    // for (auto *accel : onnx_mlir::accel::Accelerator::getAccelerators()) {
    //   // The accelerator knows whether `tensorType` is its target or not to
    //   // decide the alignment.
    //   // -1 means the accelerator does not have a specific alignment.
    //   alignment = accel->getDefaultAllocAlignment(tensorType);
    //   if (alignment != -1)
    //     break;
    // }
  }
  return alignment;
}

}
} // namespace
} // namespace
// } // namespace onnx_mlir
