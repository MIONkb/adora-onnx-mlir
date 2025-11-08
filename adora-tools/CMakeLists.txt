message(STATUS "Using ADORAConfig.cmake in: ${ADORA_INSTALL_DIR}/lib/cmake/ADORA") 
find_package(ADORA REQUIRED CONFIG)

message(STATUS "install:  ${CMAKE_BINARY_DIR}/adora-tools ") 

install(DIRECTORY include
  DESTINATION  ${CMAKE_BINARY_DIR}/adora-tools
  FILES_MATCHING
    PATTERN "*.h"  PATTERN "*.hh"  PATTERN "*.hpp"  PATTERN "*.h.inc"
    PATTERN "CMakeLists.txt" EXCLUDE
    PATTERN "*.td"           EXCLUDE
    PATTERN "*.td.inc"       EXCLUDE
    PATTERN "*.cmake"        EXCLUDE
    PATTERN "*.in"           EXCLUDE
)

include_directories(${CMAKE_BINARY_DIR}/adora-tools/include)
include_directories(${CMAKE_BINARY_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)
include_directories(include)

add_subdirectory(tools)
add_subdirectory(lib)
add_subdirectory(include)
