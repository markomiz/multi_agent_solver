cmake_minimum_required(VERSION 3.10)
project(MultiAgentSolver)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Default to Debug if no build type is specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Setup compiler flags based on the build type
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  message(STATUS "Configuring Debug build with AddressSanitizer")
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -g -fno-omit-frame-pointer -O0 -ggdb")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address")
  endif()
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
  message(STATUS "Configuring Release build with -O3")
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
  endif()
endif()

# Common flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -std=c++17")

# Find dependencies
find_package(Eigen3 REQUIRED)
find_package(osqp REQUIRED)
find_package(OsqpEigen REQUIRED)

option(USE_MKL "Use Intel MKL for multithreading" OFF )

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(USE_MKL)
        message(STATUS "Using Intel MKL for multithreading")
        find_package(MKL REQUIRED)
        add_definitions(-DEIGEN_USE_MKL)
        set(THREADING_LIBS ${MKL_LIBRARIES})
    else()
        message(STATUS "Using OpenMP for multithreading")
        find_package(OpenMP REQUIRED)
        add_definitions(-DEIGEN_USE_THREADS)
        set(THREADING_LIBS OpenMP::OpenMP_CXX)
    endif()
else()
    set(THREADING_LIBS "") # Ensure THREADING_LIBS is set for non-release builds
endif()

# Source files
file(GLOB_RECURSE SOURCES "src/*.cpp" "solvers/*.cpp" "examples/*.cpp")

# Create the executable
add_executable(dynamic_solver ${SOURCES})

# Include directories
target_include_directories(dynamic_solver PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include 
    ${CMAKE_CURRENT_SOURCE_DIR}/examples
)



# Link libraries
if(THREADING_LIBS)
    target_link_libraries(dynamic_solver Eigen3::Eigen osqp OsqpEigen::OsqpEigen ${THREADING_LIBS})
else()
    target_link_libraries(dynamic_solver Eigen3::Eigen osqp OsqpEigen::OsqpEigen)
endif()
