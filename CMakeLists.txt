cmake_minimum_required(VERSION 3.10)
project(gst-liveportrait LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86)

# Find GStreamer
find_package(PkgConfig REQUIRED)
pkg_check_modules(GST REQUIRED gstreamer-1.0 gstreamer-video-1.0 gstreamer-base-1.0)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Find TensorRT
set(TENSORRT_INCLUDE_DIR "/usr/include/x86_64-linux-gnu")
set(TENSORRT_LIB_DIR "/usr/lib/x86_64-linux-gnu")

include_directories(
    ${GST_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

link_directories(
    ${GST_LIBRARY_DIRS}
    ${TENSORRT_LIB_DIR}
)

# Plugin target
add_library(gstliveportrait SHARED
    src/gstliveportrait.cpp
    src/cuda_memory_manager.cpp
    src/trt_wrapper.cpp
    src/liveportrait_pipeline.cpp
    src/image_proc.cu
)

set_target_properties(gstliveportrait PROPERTIES 
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)

target_link_libraries(gstliveportrait
    ${GST_LIBRARIES}
    ${OpenCV_LIBS}
    CUDA::cudart
    nvinfer
    nvinfer_plugin
    stdc++fs
)

# Test executables
add_executable(test_memory
    src/test_memory.cpp
    src/cuda_memory_manager.cpp
)
target_link_libraries(test_memory CUDA::cudart)

add_executable(test_trt
    src/test_trt.cpp
    src/trt_wrapper.cpp
)
target_link_libraries(test_trt 
    CUDA::cudart 
    nvinfer 
    nvinfer_plugin
    ${OpenCV_LIBS}
    ${CMAKE_DL_LIBS}
)
