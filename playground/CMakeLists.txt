
cmake_minimum_required(VERSION 2.8)
set(OpenCV_DIR "/tmp/yaourt-tmp-yan/pkg/opencv2/usr/share/opencv")
project( DisplayImage )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( DisplayImage test.cpp )
target_link_libraries( DisplayImage ${OpenCV_LIBS} )
