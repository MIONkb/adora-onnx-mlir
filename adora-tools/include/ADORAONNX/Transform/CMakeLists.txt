set(LLVM_TARGET_DEFINITIONS Passes.td)

mlir_tablegen(Passes.h.inc -gen-pass-decls -name ADORAONNXTransformPasses)

add_public_tablegen_target(ADORAONNXTransformIncGen)

add_mlir_doc(Passes ADORAONNXTransformPasses ./ -gen-pass-doc)
