cmake_minimum_required(VERSION 3.8)
project(gpumat_with_npp LANGUAGES CUDA CXX)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

if(CUDA_FOUND)
    find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
    find_library(NPPC_LIBRARY nppc ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
    find_library(NPPIF_LIBRARY nppif ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
    if(OpenCV_FOUND)
        include_directories(${CUDA_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
        add_executable(${PROJECT_NAME} main.cpp)
        target_link_libraries(
            ${PROJECT_NAME} 
            ${CUDART_LIBRARY} 
            ${NPPC_LIBRARY} 
            ${NPPIF_LIBRARY} 
            ${OpenCV_LIBS})
    endif(OpenCV_FOUND)
endif(CUDA_FOUND)