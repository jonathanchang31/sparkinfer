cmake_minimum_required(VERSION 3.20)
project(blackwell_kernels LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# sm_89 = Ada (RTX 4090 / RTX 5090 compat)
# sm_90 = Hopper (H100)
# sm_100 = Blackwell (RTX Spark, RTX 5090, GB200)
set(CMAKE_CUDA_ARCHITECTURES "89;90;100" CACHE STRING "CUDA architectures")

option(BUILD_TESTS "Build unit tests" ON)
option(BUILD_BENCHMARKS "Build benchmarks" ON)
option(USE_CUTLASS "Link against CUTLASS headers" ON)

include(FetchContent)
if(USE_CUTLASS)
    FetchContent_Declare(
        cutlass
        GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git
        GIT_TAG main
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(cutlass)
endif()

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

include_directories(include)

# Per-domain kernel libraries
add_subdirectory(csrc/cuda/attention)
add_subdirectory(csrc/cuda/gemm)
add_subdirectory(csrc/cuda/moe)
add_subdirectory(csrc/cuda/quant)
add_subdirectory(csrc/cuda/fused)

# Unified kernel lib
add_library(blackwell_kernels INTERFACE)
target_link_libraries(blackwell_kernels INTERFACE
    bw_attn bw_gemm bw_moe bw_quant bw_fused
)
target_include_directories(blackwell_kernels INTERFACE include)

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()
