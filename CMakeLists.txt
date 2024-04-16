cmake_minimum_required(VERSION 3.12)
project(shaysweep)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_language(CUDA)

find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)
# Add Google Test
# enable_testing()
# find_package(GTest REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
set(HEADERS include/redistance.h)
set(SOURCES src/redistance.cpp)
set(BINDINGS src/bindings.cpp)
set(cuda_SOURCE src/fastsweep.cu)
pybind11_add_module(shaysweep ${HEADERS} ${SOURCES} ${cuda_SOURCE} ${BINDINGS})
target_compile_definitions(shaysweep
                           PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
target_link_libraries(shaysweep PRIVATE Eigen3::Eigen)


if(MSVC) 
	set_property(TARGET shaysweep PROPERTY VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    
    set_property(TARGET shaysweep PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    set_property(TARGET shaysweep PROPERTY CUDA_ARCHITECTURES 86)
endif()


# Add test executables
add_executable(test1 tests/test1.cpp ${HEADERS} ${SOURCES} ${cuda_SOURCE})  # Add your test source files
target_link_libraries(test1 PRIVATE Eigen3::Eigen pybind11::module)

