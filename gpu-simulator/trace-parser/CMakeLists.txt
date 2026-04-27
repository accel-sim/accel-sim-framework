FILE(GLOB files *.cc)

include_directories($ENV{CUDA_INSTALL_PATH}/include)
include_directories($ENV{GPGPUSIM_ROOT}/libcuda)
include_directories($ENV{GPGPUSIM_ROOT}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../util/tracer_nvbit/tracer_tool)

add_library(trace-parser STATIC ${files})

find_library(ZSTD_LIB zstd)
if(ZSTD_LIB)
  target_link_libraries(trace-parser ${ZSTD_LIB})
else()
  target_link_libraries(trace-parser zstd)
endif()