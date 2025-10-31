cmake_minimum_required(VERSION 3.18)
# project(gpu_optical_flow LANGUAGES CXX CUDA)
project(optflow LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES "all")

find_package(OpenCV REQUIRED)
message(STATUS "OpenCV include: ${OpenCV_INCLUDE_DIRS}")

include_directories(${OpenCV_INCLUDE_DIRS} src)

add_executable(optflow
    src/main.cpp
    src/cpu_flow.cpp
    src/utils.cpp
    # src/gpu_flow.cu
)

target_link_libraries(optflow ${OpenCV_LIBS})
set_target_properties(optflow PROPERTIES CUDA_SEPARABLE_COMPILATION ON)