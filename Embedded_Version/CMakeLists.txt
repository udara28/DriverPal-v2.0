cmake_minimum_required(VERSION 3.5)

project(DriverPal VERSION 2.0 LANGUAGES C CXX)

find_package(OpenCV COMPONENTS opencv_core opencv_video opencv_videoio opencv_imgproc opencv_imgcodecs opencv_highgui)

message(STATUS "CMAKE_SYSROOT is ${CMAKE_SYSROOT}")

include_directories(
  "${CMAKE_SYSROOT}/usr/include/drm/"
  "${CMAKE_SYSROOT}/usr/include/vitis/ai"
  "${CMAKE_SYSROOT}/opt/xilinx/xrt/include"
  "${CMAKE_SYSROOT}/usr/include/xrt"
)

link_directories(
    "${CMAKE_SYSROOT}/opt/xilinx/xrt/lib"
)

add_executable(main_app ${CMAKE_CURRENT_SOURCE_DIR}/main_app.cpp)

target_link_libraries(main_app ${OpenCV} json-c glog pthread vart-runner xnnpp-xnnpp xrt_core xrt_coreutil xilinxopencl vitis_ai_library-yolov3 vitis_ai_library-model_config)
