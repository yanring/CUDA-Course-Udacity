
cmake_minimum_required(VERSION 2.8)
set(OpenCV_DIR "/mnt/lustre/share/opencv-2.4.13")
project( DisplayImage )
FIND_PACKAGE(OpenCV REQUIRED )
FIND_PACKAGE(CUDA REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( DisplayImage test.cpp )
target_link_libraries( DisplayImage ${OpenCV_LIBS} )
