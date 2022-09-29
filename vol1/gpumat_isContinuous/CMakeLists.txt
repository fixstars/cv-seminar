cmake_minimum_required(VERSION 3.8)
project(gpumat_isContinuous)

find_package(OpenCV REQUIRED)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

if(OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
    add_executable(${PROJECT_NAME} main.cpp)
    target_link_libraries(
        ${PROJECT_NAME} 
        ${OpenCV_LIBS})
endif(OpenCV_FOUND)