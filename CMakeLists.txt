project( CLM )
cmake_minimum_required(VERSION 3.2)

find_package( OpenCV REQUIRED )
include_directories(include
                    ${OpenCV_INCLUDE_DIRS})


add_library(CLMLib 
				src/appearance/appearance.cpp
				src/clm/clm.cpp
				src/clm/detector.cpp
				src/clm/patch.cpp
				src/clm/shape.cpp
				src/io/face_data.cpp
				src/io/fps.cpp)


add_executable( clm     src/main.cpp)
add_executable( train   src/train.cpp)
add_executable( yukiti  src/yukiti.cpp)
add_executable( yukiti2 src/yukiti2.cpp )

target_link_libraries( clm      ${OpenCV_LIBS} CLMLib)
target_link_libraries( train    ${OpenCV_LIBS} CLMLib)
target_link_libraries( yukiti   ${OpenCV_LIBS} CLMLib)
target_link_libraries( yukiti2  ${OpenCV_LIBS} CLMLib)




