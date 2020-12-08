cmake_minimum_required(VERSION 3.18.1)
project(estimate-distance)
set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(SOURCE_FILES_SGBM 
    src/cpp/estimateSGBM.cpp)

set(SOURCE_FILES_BM
    src/cpp/estimateBM.cpp)

set(SOURCE_FILES_BM_CUDA
    src/cpp/estimateBM_Cuda.cpp)

add_executable(estimate-distanceSGBM ${SOURCE_FILES_SGBM})
add_executable(estimate-distanceBM ${SOURCE_FILES_BM})
add_executable(estimate-distanceBM_Cuda ${SOURCE_FILES_BM_CUDA})

target_link_libraries(estimate-distanceSGBM ${OpenCV_LIBS})
target_link_libraries(estimate-distanceBM ${OpenCV_LIBS})
target_link_libraries(estimate-distanceBM_Cuda ${OpenCV_LIBS})
