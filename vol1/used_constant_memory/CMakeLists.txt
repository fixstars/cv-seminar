cmake_minimum_required(VERSION 3.8)
project(used_constant_memory LANGUAGES CUDA CXX)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

file(GLOB SOURCES "*.cu" "*.cpp" "*.c" "cuh" "*.h" "*.hpp")

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_COMPILER_IS_CLANGCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -march=native")
endif()

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

if(CUDA_FOUND)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-Xcompiler -std=c++11")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-gencode arch=compute_75,code=sm_75")

    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-G;-g;-lineinfo;--ptxas-options=-v")
    else()
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-O3;-g;-lineinfo;--ptxas-options=-v")
    endif(CMAKE_BUILD_TYPE STREQUAL "Debug")

    if(OpenCV_FOUND)
        include_directories(${OpenCV_INCLUDE_DIRS})
        add_executable(${PROJECT_NAME} ${SOURCES})
        target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
    endif(OpenCV_FOUND)
endif(CUDA_FOUND)
