cmake_minimum_required(VERSION 3.22)
project(ParCIS_MXKMeans VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

include(CTest)
option(MXKMEANS_BUILD_CUDA "Build the GPU executable and GPU tests" ON)

if(BUILD_TESTING)
    add_executable(host_checks tests/host_checks.cpp)
    target_include_directories(host_checks PRIVATE include)
    add_test(NAME host_checks COMMAND host_checks)
endif()

if(MXKMEANS_BUILD_CUDA)
    # Architecture-specific SM120 instructions require the 'a' suffix.
    set(CMAKE_CUDA_ARCHITECTURES OFF)
    set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
    enable_language(CUDA)
    find_package(CUDAToolkit 12.8 REQUIRED)
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.8)
        message(FATAL_ERROR "CUDA compiler 12.8 or newer is required")
    endif()
    set(CUTLASS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass" CACHE PATH "CUTLASS source checkout")
    if(NOT EXISTS "${CUTLASS_DIR}/include/cutlass/cutlass.h" OR
       NOT EXISTS "${CUTLASS_DIR}/tools/util/include/cutlass/util/packed_stride.hpp")
        message(FATAL_ERROR "Set -DCUTLASS_DIR=/path/to/cutlass (see README.md)")
    endif()
    add_library(mxkmeans_core STATIC
        src/kmeans.cu
        src/kernels/gemm.cu
        src/kernels/quantize.cu
        src/kernels/assignment.cu
        src/kernels/update.cu
    )
    target_include_directories(mxkmeans_core PUBLIC include "${CUTLASS_DIR}/include"
        "${CUTLASS_DIR}/tools/util/include")
    target_link_libraries(mxkmeans_core PUBLIC CUDA::cudart CUDA::cublas)
    target_compile_options(mxkmeans_core PUBLIC
        $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_120a,code=sm_120a>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
        $<$<COMPILE_LANGUAGE:CUDA>:--static-global-template-stub=true>
    )
    set_target_properties(mxkmeans_core PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
    add_executable(mxkmeans src/main.cu)
    target_link_libraries(mxkmeans PRIVATE mxkmeans_core)
    set_target_properties(mxkmeans PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
    if(BUILD_TESTING)
        add_executable(cuda_checks tests/cuda_checks.cu)
        target_link_libraries(cuda_checks PRIVATE mxkmeans_core)
        set_target_properties(cuda_checks PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
        add_test(NAME cuda_checks COMMAND cuda_checks)
        set_tests_properties(cuda_checks PROPERTIES SKIP_RETURN_CODE 77)
        add_test(NAME cli_help COMMAND mxkmeans --help)
    endif()
endif()
