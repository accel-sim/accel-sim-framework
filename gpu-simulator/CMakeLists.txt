cmake_minimum_required(VERSION 3.17)

# Project name and version
project(Accel-Sim
        VERSION 1.2.0 
        DESCRIPTION "Accel-Sim" 
        HOMEPAGE_URL https://github.com/accel-sim/accel-sim-framework
        LANGUAGES CXX)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if($ENV{ACCELSIM_CONFIG} STREQUAL "debug")
    message("string is debug")
    set(CMAKE_BUILD_TYPE Debug)
else()
    set(CMAKE_BUILD_TYPE Release)
endif()

# check envrionment variable string compare
if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    add_compile_definitions(DEBUG=1)
    add_compile_options(-Wall -O0 -g3 -fPIC)
else()
    add_compile_definitions(DEBUG=0)
    add_compile_options(-Wall -O3 -g3 -fPIC)
endif()

# run command
execute_process(
    COMMAND git log --abbrev-commit -n 1
    COMMAND head -1
    COMMAND sed -re "s/commit (.*)/\\1/"
    OUTPUT_VARIABLE GIT_COMMIT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    )
execute_process(
    COMMAND git diff --numstat
    COMMAND wc
    COMMAND sed -re "s/^\\s+([0-9]+).*/\\1./"
    OUTPUT_VARIABLE GIT_FILES_CHANGED_A
    OUTPUT_STRIP_TRAILING_WHITESPACE
    )
execute_process(
    COMMAND git diff --numstat --cached
    COMMAND wc
    COMMAND sed -re "s/^\\s+([0-9]+).*/\\1/"
    OUTPUT_VARIABLE GIT_FILES_CHANGED
    OUTPUT_STRIP_TRAILING_WHITESPACE
    )
execute_process(
    COMMAND date --iso-8601=minutes
    OUTPUT_VARIABLE TIME
    OUTPUT_STRIP_TRAILING_WHITESPACE
    )

set(ACCELSIM_BUILD accelsim-commit-${GIT_COMMIT}_modified_${GIT_FILES_CHANGED_A}${GIT_FILES_CHANGED}_${TIME})

file(WRITE ${CMAKE_BINARY_DIR}/accelsim_version.h "const char *g_accelsim_version=\"${ACCELSIM_BUILD}\";")

add_subdirectory($ENV{GPGPUSIM_ROOT})
add_subdirectory(trace-driven)
add_subdirectory(trace-parser)


include_directories($ENV{CUDA_INSTALL_PATH}/include)
include_directories($ENV{GPGPUSIM_ROOT}/libcuda)
include_directories($ENV{GPGPUSIM_ROOT}/src)
include_directories(${CMAKE_BINARY_DIR})
include_directories(ISA_Def)
include_directories(trace-driven)
include_directories(trace-parser)


add_executable(accel-sim.out main.cc)
# target_link_libraries(accel-sim.out PUBLIC cudart)
target_link_libraries(accel-sim.out PUBLIC cuda ptxsim gpgpusim intersim accelwattch entrypoint)
target_link_libraries(accel-sim.out PUBLIC -lm -lz -lGL -pthread)
target_link_libraries(accel-sim.out PUBLIC trace-driven trace-parser)
# target_link_libraries(accel-sim.out PUBLIC cudart)

install(TARGETS accel-sim.out DESTINATION ${CMAKE_SOURCE_DIR}/bin/$ENV{ACCELSIM_CONFIG})