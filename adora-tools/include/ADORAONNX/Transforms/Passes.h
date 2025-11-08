//===- Passes.h - Pass Entrypoints ------------------------------*- C++ -*-===//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose pass constructors.
//
//===----------------------------------------------------------------------===//

#ifndef ADORAONNX_TRANSORM_PASSES_H_
#define ADORAONNX_TRANSORM_PASSES_H_

#include "ADORA/Dialect/ADORATensor/IR/ADORATensor.h"
#include "mlir/Pass/Pass.h"
namespace mlir {
namespace ADORA {
namespace ADORATensor{

/// Lower some operators in Linalg dialect to 
#define GEN_PASS_DEF_OUTLINEONNXLAYERSPASS
std::unique_ptr<OperationPass<ModuleOp>> createOutlineONNXLayersPass();
// std::unique_ptr<OperationPass<ModuleOp>> createADORATensorOpCdfgGenPass();
// std::unique_ptr<OperationPass<ModuleOp>> createADORAGemmOpStrategyDecisionPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//
// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#define GEN_PASS_DECL
#define GEN_PASS_CLASSES
#include "ADORAONNX/Transforms/Passes.h.inc"

} // namespace ADORATensor
} // namespace ADORA
} // namespace mlir

#endif // ADORATensor_DIALECT_PASSES_H_
