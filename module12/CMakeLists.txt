cmake_minimum_required(VERSION 3.10)
project(convolution LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Find OpenCL
find_package(OpenCL REQUIRED)

# Add executable
add_executable(convolution src/main.cpp)

# Link libraries
if(OpenCV_FOUND)
    target_include_directories(convolution PRIVATE ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(convolution PRIVATE ${OpenCV_LIBS})
endif()

if(OpenCL_FOUND)
    target_include_directories(convolution PRIVATE ${OpenCL_INCLUDE_DIRS})
    target_link_libraries(convolution PRIVATE ${OpenCL_LIBRARIES})
endif()
