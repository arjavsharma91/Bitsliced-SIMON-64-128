cmake_minimum_required(VERSION 3.14)
project(SimonBitsliceAVX2 CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    add_compile_options(/O2 /arch:AVX2)
else()
    add_compile_options(-O3 -march=native -mavx2)
endif()

add_library(simon_engine INTERFACE)
target_include_directories(simon_engine INTERFACE 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

add_executable(simon_test main.cpp)
target_link_libraries(simon_test PRIVATE simon_engine)
