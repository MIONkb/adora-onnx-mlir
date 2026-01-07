//===- Passes.h - Pass Entrypoints ------------------------------*- C++ -*-===//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose pass constructors.
//
//===----------------------------------------------------------------------===//

#ifndef ADORAONNX_TRANSFORM_PASSES_H_
#define ADORAONNX_TRANSFORM_PASSES_H_

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace ADORA {
namespace ADORATensor {

#define GEN_PASS_DEF_SETONNXBATCHSIZEPASS
std::unique_ptr<OperationPass<ModuleOp>> createSetONNXBatchSizePass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//
#define GEN_PASS_REGISTRATION
#define GEN_PASS_DECL
#define GEN_PASS_CLASSES
#include "ADORAONNX/Transform/Passes.h.inc"

} // namespace ADORATensor
} // namespace ADORA
} // namespace mlir

#endif // ADORAONNX_TRANSFORM_PASSES_H_
