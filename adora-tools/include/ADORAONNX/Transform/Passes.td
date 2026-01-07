//===-- Passes.td - ADORAONNX transform pass definition file ----*- tablegen -*-===//
//===----------------------------------------------------------------------===//
//
// Defines the ADORAONNX transform passes.
//
//===----------------------------------------------------------------------===//
#ifndef ADORAONNX_TRANSFORM_PASSES_TLBGEN
#define ADORAONNX_TRANSFORM_PASSES_TLBGEN

include "mlir/IR/BuiltinDialect.td"
include "mlir/Pass/PassBase.td"

def SetONNXBatchSizePass : Pass<"adora-onnx-set-batch", "ModuleOp"> {
  let summary = "Set the batch dimension on entry-point tensor types";
  let description = [{
    This pass updates the first dimension of ranked tensor types for entry-point
    function arguments so they use the specified batch size.
  }];
  let constructor = "createSetONNXBatchSizePass()";
  let options = [
    Option<"batchSize", "batch-size", "int", /*default=*/"1",
           "Batch size to set on the first dimension of input tensors">,
  ];
}

#endif // ADORAONNX_TRANSFORM_PASSES_TLBGEN
