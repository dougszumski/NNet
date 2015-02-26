cmake_minimum_required (VERSION 2.8)
project (NNet)

SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})

include_directories(src)

set (CMAKE_C_FLAGS "-Wall -std=c99 -O2")

set(LIB_SRC
   ${PROJECT_SOURCE_DIR}/src/nnet.c
   ${PROJECT_SOURCE_DIR}/src/loader.c
   ${PROJECT_SOURCE_DIR}/src/math_utils.c
)

add_library(nnet STATIC ${LIB_SRC})

set(MAIN_SRC
   ${PROJECT_SOURCE_DIR}/src/main.c
)

add_executable(run ${MAIN_SRC})
target_link_libraries (run nnet gsl gslcblas m)

set (CMAKE_CXX_FLAGS "-Wall -std=c++11")

set(TEST_SRC
   ${PROJECT_SOURCE_DIR}/src/tests.cpp
)

add_executable(tests ${TEST_SRC})
target_link_libraries (tests nnet gsl gslcblas m)

