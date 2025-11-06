# SPDX-License-Identifier: Apache-2.0


message(STATUS "Using ADORAConfig.cmake in: ${ADORA_INSTALL_DIR}/lib/cmake/ADORA") 
find_package(ADORA REQUIRED CONFIG)
# include_directories(${ADORA_INSTALL_DIR}/include)
add_onnx_mlir_executable(adora-onnx-opt
  adora-onnx-opt.cpp
  RegisterPasses.cpp

  LINK_LIBS PRIVATE
  OMCompilerDialects
  OMCompilerOptions
  OMCompilerPasses
  # OMAccelerator
  # OMVersion
  # MLIROpenMPToLLVM
  MLIRAffineTransforms
  MLIRLinalgTransforms
  MLIRMemRefTransforms
  MLIROpenMPToLLVM
  MLIROptLib
  # MLIRSCFToOpenMP

  # ADORATensor
  ADORATensorOps
  )
  
install(TARGETS adora-onnx-opt)