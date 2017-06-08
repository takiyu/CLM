cmake_minimum_required(VERSION 3.2)

project(CLM CXX C)
set (CMAKE_CXX_STANDARD 98)  # C++ 98

# Set default build type
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Set output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Opencv
find_package( OpenCV REQUIRED )
include_directories(${OpenCV_INCLUDE_DIRS})

# libCLM.so
add_library(CLM
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

target_link_libraries(clm      ${OpenCV_LIBS} CLM)
target_link_libraries(train    ${OpenCV_LIBS} CLM)
target_link_libraries(yukiti   ${OpenCV_LIBS} CLM)
target_link_libraries(yukiti2  ${OpenCV_LIBS} CLM)
