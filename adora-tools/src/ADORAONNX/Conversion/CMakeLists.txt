# To solve error: exception handling disabled, use '-fexceptions' to enable
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSPDLOG_NO_EXCEPTIONS ")
set(SPDLOG_NO_EXCEPTIONS ON)

add_mlir_dialect_library(ADORAONNXConversion
  ConvertONNXLayersToAdora.cpp
  ONNXToADORACommon.cpp

  OpConversion/Outline.cpp
  OpConversion/AdoraONNXTransposeLowering.cpp
  OpConversion/AdoraKrnlGlobalOpLowering.cpp
  OpConversion/AdoraONNXOpConversion.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_BINARY_DIR}/adora-tools/include 

  DEPENDS
  ADORAONNXConversionIncGen
  
  LINK_LIBS PUBLIC
  GeneralCDFGGen
  # ADORAONNXTransformIncGen
  # MLIRADORALoopTransforms
  ADORATensorOps
  MLIRADORAKernelOp
  # ADORATensorLower
  ADORATensorUtility
  # MLIRTensorDialect

  OMONNXToKrnl
  OMKrnlToAffine
)

set(LIBS
  ADORAONNXTransforms
)

