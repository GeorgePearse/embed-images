cmake_minimum_required(VERSION 3.18)
project(embed-images LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── CUDA + TensorRT ──────────────────────────────────────────────
find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)

# TensorRT — try standard paths, allow override via -DTENSORRT_ROOT=...
if(NOT TENSORRT_ROOT)
    set(TENSORRT_ROOT "/usr" CACHE PATH "TensorRT install prefix")
endif()

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS ${TENSORRT_ROOT}/include /usr/include/x86_64-linux-gnu)
find_library(NVINFER_LIB nvinfer
    HINTS ${TENSORRT_ROOT}/lib ${TENSORRT_ROOT}/lib/x86_64-linux-gnu)
find_library(NVONNXPARSER_LIB nvonnxparser
    HINTS ${TENSORRT_ROOT}/lib ${TENSORRT_ROOT}/lib/x86_64-linux-gnu)

if(NOT NVINFER_LIB OR NOT NVONNXPARSER_LIB)
    message(FATAL_ERROR "TensorRT not found. Set -DTENSORRT_ROOT=<path>")
endif()
message(STATUS "TensorRT include: ${TENSORRT_INCLUDE_DIR}")
message(STATUS "nvinfer: ${NVINFER_LIB}")
message(STATUS "nvonnxparser: ${NVONNXPARSER_LIB}")

# ── OpenCV ───────────────────────────────────────────────────────
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

# ── Header-only deps via FetchContent ────────────────────────────
include(FetchContent)

FetchContent_Declare(cli11
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
    GIT_TAG        v2.4.2)

FetchContent_Declare(json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3)

FetchContent_Declare(spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG        v1.15.3)

FetchContent_Declare(cnpy
    GIT_REPOSITORY https://github.com/rogersce/cnpy.git
    GIT_TAG        master)

FetchContent_MakeAvailable(cli11 json spdlog cnpy)

# ── Build ────────────────────────────────────────────────────────
add_executable(embed-images
    src/main.cpp
    src/engine.cpp
    src/thumbnail.cpp
    src/inference.cpp
    src/output.cpp
    src/duplicates.cpp
)

target_include_directories(embed-images PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${TENSORRT_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
    ${cnpy_SOURCE_DIR}
)

target_link_libraries(embed-images PRIVATE
    ${NVINFER_LIB}
    ${NVONNXPARSER_LIB}
    CUDA::cudart
    ${OpenCV_LIBS}
    CLI11::CLI11
    nlohmann_json::nlohmann_json
    spdlog::spdlog
    cnpy
)
