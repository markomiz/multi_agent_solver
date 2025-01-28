cmake_minimum_required(VERSION 3.10)

# Project name
project(DynamicSolver)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-std=c++17 -Wall -Wextra -O3")

# Add Eigen and OSQP-Eigen libraries
find_package(Eigen3 REQUIRED)
find_package(osqp REQUIRED)
find_package(OsqpEigen REQUIRED)

# Specify source files
add_executable(dynamic_solver src/main.cpp)

# Include directories
target_include_directories(dynamic_solver PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Link Eigen and OSQP-Eigen libraries
target_link_libraries(dynamic_solver Eigen3::Eigen osqp OsqpEigen::OsqpEigen)

# Optional: Enable warnings
target_compile_options(dynamic_solver PRIVATE -Wall -Wextra -Wpedantic)
