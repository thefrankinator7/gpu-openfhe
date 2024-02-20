cmake_minimum_required(VERSION 3.24)
project(gpu_funcs)

enable_language(HIP)
set (SOURCE_FILES gpu_functions.hip gpu_functions.h)
add_library(gpu_funcs STATIC ${SOURCE_FILES})
target_compile_options(gpu_funcs PUBLIC -fPIC)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

