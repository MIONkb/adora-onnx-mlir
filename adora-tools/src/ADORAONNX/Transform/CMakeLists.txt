add_mlir_dialect_library(ADORAONNXTransforms
  SetBatchSize.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_BINARY_DIR}/adora-tools/include

  DEPENDS
  ADORAONNXTransformIncGen

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRIR
)
