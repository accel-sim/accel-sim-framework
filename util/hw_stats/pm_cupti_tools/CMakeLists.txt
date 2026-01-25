cmake_minimum_required(VERSION 3.18)
project(pm_cupti_tools LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA Toolkit (provides CUPTI paths)
find_package(CUDAToolkit REQUIRED)

# CUPTI paths - use CUDAToolkit_ROOT which points to CUDA install root
set(CUPTI_ROOT "${CUDAToolkit_BIN_DIR}/../extras/CUPTI")
set(CUPTI_INCLUDE_DIR "${CUPTI_ROOT}/include")
set(CUPTI_LIB_DIR "${CUPTI_ROOT}/lib64")
set(CUPTI_SAMPLES_COMMON "${CUPTI_ROOT}/samples/common")

# # Verify CUPTI exists
# if(NOT EXISTS "${CUPTI_INCLUDE_DIR}/cupti.h")
#     message(FATAL_ERROR "CUPTI not found at ${CUPTI_INCLUDE_DIR}. "
#         "Please ensure CUDA Toolkit is installed with CUPTI support.")
# endif()

# # Verify helper_cupti.h exists
# if(NOT EXISTS "${CUPTI_SAMPLES_COMMON}/helper_cupti.h")
#     message(FATAL_ERROR "CUPTI samples common not found at ${CUPTI_SAMPLES_COMMON}. "
#         "Please ensure CUDA Toolkit samples are installed.")
# endif()

# Find CUPTI library (may be in CUPTI/lib64 or main CUDA lib64)
find_library(CUPTI_LIBRARY
    NAMES cupti
    PATHS ${CUPTI_LIB_DIR} ${CUDAToolkit_LIBRARY_DIR}
    REQUIRED
)

message(STATUS "CUDA Toolkit: ${CUDAToolkit_BIN_DIR}")
message(STATUS "CUPTI Include: ${CUPTI_INCLUDE_DIR}")
message(STATUS "CUPTI Samples Common: ${CUPTI_SAMPLES_COMMON}")
message(STATUS "CUPTI Library: ${CUPTI_LIBRARY}")

# Injection library (shared library for CUDA_INJECTION64_PATH)
add_library(pmsampling_injection SHARED
    pm_injection.cu
    pm_sampling.cu
)

target_include_directories(pmsampling_injection PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CUPTI_INCLUDE_DIR}
    ${CUPTI_SAMPLES_COMMON}
)

target_link_libraries(pmsampling_injection PRIVATE
    ${CUPTI_LIBRARY}
    CUDA::cuda_driver
)

target_compile_options(pmsampling_injection PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -fPIC>
)

set_target_properties(pmsampling_injection PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME "pmsampling_injection"
)

# Simple test kernel for metric sweeps
add_executable(test_kernel test_kernel.cu)
target_link_libraries(test_kernel PRIVATE CUDA::cudart)

# Install targets
install(TARGETS pmsampling_injection test_kernel
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
