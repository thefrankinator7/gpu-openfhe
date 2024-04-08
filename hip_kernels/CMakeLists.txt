cmake_minimum_required(VERSION 3.24)
project(gpu_funcs)

enable_language(HIP)

include_directories(include)

set (SOURCE_FILES gpu_functions.hip gpu_functions.h)
add_library(gpu_funcs STATIC ${SOURCE_FILES})
add_executable(test_gpu test_gpu_kernels.hip gpu_functions.hip)
target_compile_options(gpu_funcs PUBLIC -fPIC)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# include_directories(include)
# target_include_directories(ntt_exe PUBLIC /opt/rocm-5.6.0/include)
# link_libraries(hip::device)
