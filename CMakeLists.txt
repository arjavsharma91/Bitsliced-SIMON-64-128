cmake_minimum_required(VERSION 3.16)
project(SimonBitsliceEngine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
    message(FATAL_ERROR "Please build out-of-source (e.g., cmake -B build && cmake --build build).")
endif()

add_executable(simon_test main.cpp)

target_include_directories(simon_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(simon_test PRIVATE
        -O3
        -mavx2
        -Wall
        -Wextra
    )
elseif(MSVC)
    target_compile_options(simon_test PRIVATE
        /O2
        /arch:AVX2
        /W4
    )
endif()
