cmake_minimum_required(VERSION 3.10)

project(toolsdetect_test LANGUAGES CXX)

# ------------------ 用户可配置项 ------------------
# 目前默认你的环境使用 MinGW（你原来写的）。如果你需要改用 MSVC, 在命令行或 VS Code CMake Kit 中选择 MSVC kit.
# 若要启用 ONNX Runtime，请在 configure 时传入 -DONNXRUNTIME_DIR="D:/onnxruntime"
# 例如:
#   cmake -B build -S . -G "MinGW Makefiles" -DONNXRUNTIME_DIR="D:/onnxruntime"
# 或 (若使用 MSVC):
#   cmake -B build -S . -G "Visual Studio 17 2022" -A x64 -DONNXRUNTIME_DIR="D:/onnxruntime"
#
# ---------------------------------------------------

# --------------（默认保留你的 MinGW 设置）-------------
# 你在原文件中显式设置了 MinGW 的编译器，这里保留（如果你使用 MSVC，请删除或覆盖）
set(CMAKE_C_COMPILER   "D:/mingw64/bin/gcc.exe" CACHE PATH "C compiler")
set(CMAKE_CXX_COMPILER "D:/mingw64/bin/g++.exe" CACHE PATH "C++ compiler")
# ------------------------------------------------------

# 指向包含 OpenCVConfig.cmake 的目录（你之前给的路径）
set(OpenCV_DIR "D:/opencv/install/x64/mingw/lib" CACHE PATH "OpenCV config dir")
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc highgui)

message(STATUS "Using OpenCV from: ${OpenCV_DIR}")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libs: ${OpenCV_LIBS}")

# 源文件列表：将 yoloinfer.cpp 加入
set(SRC_FILES
    src/main.cpp
    src/auth.cpp
    src/logger.cpp
    src/detector.cpp
    src/inventory_compare.cpp
    src/yoloinfer.cpp        # <-- 新增：推理实现文件
    src/yoloinfer.h         # <-- 新增：建议把头文件加入仓库
    # src/vision_pipeline.cpp
)

add_executable(${PROJECT_NAME} ${SRC_FILES})

target_include_directories(${PROJECT_NAME} PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# 链接 OpenCV
target_link_libraries(${PROJECT_NAME} PRIVATE
    ${OpenCV_LIBS}
)

# 如果外面没传 ONNXRUNTIME_DIR，就默认用 E:/onnxruntime-win-x64-gpu-1.23.2
if(NOT DEFINED ONNXRUNTIME_DIR)
    set(ONNXRUNTIME_DIR "E:/onnxruntime-win-x64-gpu-1.23.2" CACHE PATH "ONNX Runtime base dir" FORCE)
endif()

# ---------------- ONNX Runtime 支持（可选） ----------------
# 使用方法：cmake -D ONNXRUNTIME_DIR="D:/onnxruntime" ...
if(DEFINED ONNXRUNTIME_DIR)
    message(STATUS "ONNXRUNTIME_DIR specified: ${ONNXRUNTIME_DIR}")
    set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_DIR}/include" CACHE PATH "onnxruntime include")
    set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_DIR}/lib" CACHE PATH "onnxruntime lib")
    set(ONNXRUNTIME_BIN_DIR "${ONNXRUNTIME_DIR}/bin" CACHE PATH "onnxruntime bin")

    # add include path
    if(EXISTS "${ONNXRUNTIME_INCLUDE_DIR}")
        message(STATUS "ONNX Runtime include dir: ${ONNXRUNTIME_INCLUDE_DIR}")
        target_include_directories(${PROJECT_NAME} PRIVATE "${ONNXRUNTIME_INCLUDE_DIR}")
    else()
        message(WARNING "ONNXRUNTIME include dir not found: ${ONNXRUNTIME_INCLUDE_DIR}")
    endif()

    # Try to find onnxruntime import lib (MSVC style .lib) first
    set(FOUND_ONNXRUNTIME_LIB OFF)
    if(EXISTS "${ONNXRUNTIME_LIB_DIR}/onnxruntime.lib")
        message(STATUS "Found onnxruntime.lib (MSVC import lib) at ${ONNXRUNTIME_LIB_DIR}/onnxruntime.lib")
        add_library(OnnxRuntimeImported STATIC IMPORTED)
        set_target_properties(OnnxRuntimeImported PROPERTIES
            IMPORTED_LOCATION "${ONNXRUNTIME_LIB_DIR}/onnxruntime.lib"
            INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}"
        )
        target_link_libraries(${PROJECT_NAME} PRIVATE OnnxRuntimeImported)
        set(FOUND_ONNXRUNTIME_LIB ON)
    endif()

    # If MSVC lib not found, try to find mingw import lib or .a (libonnxruntime.a or onnxruntime.lib)
    if(NOT FOUND_ONNXRUNTIME_LIB)
        # possible names: libonnxruntime.a or onnxruntime.a
        if(EXISTS "${ONNXRUNTIME_LIB_DIR}/libonnxruntime.a")
            message(STATUS "Found libonnxruntime.a at ${ONNXRUNTIME_LIB_DIR}/libonnxruntime.a (MinGW import lib)")
            add_library(OnnxRuntimeImported STATIC IMPORTED)
            set_target_properties(OnnxRuntimeImported PROPERTIES
                IMPORTED_LOCATION "${ONNXRUNTIME_LIB_DIR}/libonnxruntime.a"
                INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}"
            )
            target_link_libraries(${PROJECT_NAME} PRIVATE OnnxRuntimeImported)
            set(FOUND_ONNXRUNTIME_LIB ON)
        elseif(EXISTS "${ONNXRUNTIME_LIB_DIR}/onnxruntime.a")
            message(STATUS "Found onnxruntime.a at ${ONNXRUNTIME_LIB_DIR}/onnxruntime.a (MinGW import lib)")
            add_library(OnnxRuntimeImported STATIC IMPORTED)
            set_target_properties(OnnxRuntimeImported PROPERTIES
                IMPORTED_LOCATION "${ONNXRUNTIME_LIB_DIR}/onnxruntime.a"
                INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}"
            )
            target_link_libraries(${PROJECT_NAME} PRIVATE OnnxRuntimeImported)
            set(FOUND_ONNXRUNTIME_LIB ON)
        endif()
    endif()

    # If still not found, try to link against DLL directly using an IMPORTED SHARED target (for runtime load)
    if(NOT FOUND_ONNXRUNTIME_LIB)
        # check for onnxruntime.dll inside bin
        if(EXISTS "${ONNXRUNTIME_BIN_DIR}/onnxruntime.dll")
            message(STATUS "Found onnxruntime.dll at ${ONNXRUNTIME_BIN_DIR}/onnxruntime.dll — will create IMPORTED library that references the DLL (may require an import lib for linking)")
            add_library(OnnxRuntimeDLL SHARED IMPORTED GLOBAL)
            set_target_properties(OnnxRuntimeDLL PROPERTIES
                IMPORTED_LOCATION "${ONNXRUNTIME_BIN_DIR}/onnxruntime.dll"
                INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}"
            )
            # try link by name; this may or may not work depending on toolchain
            target_link_libraries(${PROJECT_NAME} PRIVATE OnnxRuntimeDLL)
            set(FOUND_ONNXRUNTIME_LIB ON)
            # ensure DLL copied to output after build
            add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${ONNXRUNTIME_BIN_DIR}/onnxruntime.dll"
                $<TARGET_FILE_DIR:${PROJECT_NAME}>
            )
        endif()
    endif()

    if(NOT FOUND_ONNXRUNTIME_LIB)
        message(WARNING "ONNX Runtime library not found in ${ONNXRUNTIME_LIB_DIR} nor bin. \
If you are using MinGW, note: official prebuilt onnxruntime.lib is usually MSVC format and not directly linkable by MinGW. \
Either build ONNX Runtime for MinGW or switch to MSVC toolchain. To disable ONNX integration, reconfigure without -DONNXRUNTIME_DIR.")
    else()
        message(STATUS "ONNX Runtime integration configured.")
    endif()
else()
    message(STATUS "ONNXRUNTIME_DIR not specified — ONNX integration disabled.")
endif()

# --------------- 跨平台 / 输出目录设置（可选） ---------------
# Make sure we build 64-bit if ONNX runtime is x64; user can set CMAKE_GENERATOR or -A accordingly
if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(WARNING "Build is not 64-bit. ONNX Runtime prebuilt packages are typically x64; ensure your toolchain/target matches.")
endif()

# ----------------- 编译选项 -----------------
# Example: enable warnings
if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE /W4 /WX-)
else()
    target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wno-unused-parameter)
    # Ensure C++17
    set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)
endif()

# ----------------- 最终信息 -----------------
message(STATUS "Project target: ${PROJECT_NAME}")
message(STATUS "Sources: ${SRC_FILES}")
