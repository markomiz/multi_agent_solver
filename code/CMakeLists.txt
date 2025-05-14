cmake_minimum_required(VERSION 3.14)
project(MultiAgentSolver VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Default to Debug if no build type is specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Add the header-only library
add_library(MultiAgentSolver INTERFACE)
target_include_directories(MultiAgentSolver INTERFACE 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(MultiAgentSolver INTERFACE cxx_std_17)
target_compile_options(MultiAgentSolver INTERFACE -Wall -Wextra -Wpedantic)

# Find dependencies
find_package(Eigen3 REQUIRED)
find_package(osqp REQUIRED)
find_package(OsqpEigen REQUIRED)

target_link_libraries(MultiAgentSolver INTERFACE Eigen3::Eigen osqp OsqpEigen::OsqpEigen)

# Optional multithreading
option(USE_MKL "Use Intel MKL for multithreading" OFF)

if(USE_MKL)
    message(STATUS "Using Intel MKL for multithreading")
    find_package(MKL REQUIRED)
    target_compile_definitions(MultiAgentSolver INTERFACE -DEIGEN_USE_MKL)
    target_link_libraries(MultiAgentSolver INTERFACE ${MKL_LIBRARIES})
else()
    message(STATUS "Using OpenMP for multithreading")
    find_package(OpenMP REQUIRED)
    target_compile_definitions(MultiAgentSolver INTERFACE -DEIGEN_USE_THREADS)
    target_link_libraries(MultiAgentSolver INTERFACE OpenMP::OpenMP_CXX)
endif()

# Example Executable
add_executable(dynamic_solver examples/main.cpp)
target_link_libraries(dynamic_solver PRIVATE MultiAgentSolver)
target_include_directories(dynamic_solver PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/examples)

# Installation
install(TARGETS MultiAgentSolver EXPORT MultiAgentSolverTargets)
install(EXPORT MultiAgentSolverTargets
    FILE MultiAgentSolverConfig.cmake
    NAMESPACE MultiAgentSolver::
    DESTINATION lib/cmake/MultiAgentSolver
)
install(DIRECTORY include/ DESTINATION include)
