cmake_minimum_required(VERSION 3.10)

project(toolsdetect_test LANGUAGES CXX)

# 使用 MinGW 的编译器
set(CMAKE_C_COMPILER   "D:/mingw64/bin/gcc.exe")
set(CMAKE_CXX_COMPILER "D:/mingw64/bin/g++.exe")

# 指向包含 OpenCVConfig.cmake 的目录
# 你告诉我它在 D:/opencv/install/x64/mingw/lib/OpenCVConfig.cmake
# => OpenCV_DIR 就写它所在的那个目录 D:/opencv/install/x64/mingw/lib
set(OpenCV_DIR "D:/opencv/install/x64/mingw/lib")

# 找我们要用的模块
find_package(OpenCV REQUIRED core imgcodecs imgproc highgui)

message(STATUS "Using OpenCV from: ${OpenCV_DIR}")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libs: ${OpenCV_LIBS}")

add_executable(${PROJECT_NAME}
    src/main.cpp
    src/auth.cpp
    src/logger.cpp
    src/detector.cpp
    src/inventory_compare.cpp
    # 你可以把 vision_pipeline.cpp 暂时留着或去掉
    # src/vision_pipeline.cpp
)



target_include_directories(toolsdetect_test PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(toolsdetect_test PRIVATE
    ${OpenCV_LIBS}
)
