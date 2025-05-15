cmake_minimum_required(VERSION 3.15)
project(MultiAgentSolver VERSION 1.0.0 LANGUAGES CXX)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Option to build examples
option(BUILD_EXAMPLES "Build example executables" ON)
option(BUILD_SHARED_LIBS "Build MultiAgentSolver as a shared library" OFF)

# Dependencies
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenMP)

# OSQP and OsqpEigen
find_package(osqp QUIET)
find_package(OsqpEigen QUIET)

include(FetchContent)
if(NOT osqp_FOUND)
    message(STATUS "OSQP not found, fetching via FetchContent...")
    FetchContent_Declare(
        osqp
        GIT_REPOSITORY https://github.com/osqp/osqp.git
        GIT_TAG v0.6.2
    )
    FetchContent_MakeAvailable(osqp)
endif()

if(NOT OsqpEigen_FOUND)
    message(STATUS "OsqpEigen not found, fetching via FetchContent...")
    FetchContent_Declare(
        OsqpEigen
        GIT_REPOSITORY https://github.com/robotology/osqp-eigen.git
        GIT_TAG v0.10.0
    )
    FetchContent_MakeAvailable(OsqpEigen)
endif()

# Define the header-only library
add_library(MultiAgentSolver INTERFACE)

# Include directories
target_include_directories(MultiAgentSolver INTERFACE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(MultiAgentSolver INTERFACE
    Eigen3::Eigen
    OsqpEigen::OsqpEigen
    osqp::osqp
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(MultiAgentSolver INTERFACE OpenMP::OpenMP_CXX)
endif()

# Examples
if(BUILD_EXAMPLES)
    add_executable(multi_agent_lqr examples/multi_agent_lqr.cpp)
    target_link_libraries(multi_agent_lqr PRIVATE MultiAgentSolver)

    add_executable(single_track_ocp examples/single_track_ocp.cpp)
    target_link_libraries(single_track_ocp PRIVATE MultiAgentSolver)

    add_executable(multi_agent_single_track examples/multi_agent_single_track.cpp)
    target_link_libraries(multi_agent_single_track PRIVATE MultiAgentSolver)

endif()

# Install targets
install(TARGETS MultiAgentSolver EXPORT MultiAgentSolverTargets)
install(DIRECTORY include/ DESTINATION include)

# Export package
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/MultiAgentSolverConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    "${CMAKE_SOURCE_DIR}/cmake/MultiAgentSolverConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/MultiAgentSolverConfig.cmake"
    INSTALL_DESTINATION lib/cmake/MultiAgentSolver
)

install(EXPORT MultiAgentSolverTargets
    FILE MultiAgentSolverTargets.cmake
    NAMESPACE MultiAgentSolver::
    DESTINATION lib/cmake/MultiAgentSolver
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/MultiAgentSolverConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/MultiAgentSolverConfigVersion.cmake"
    DESTINATION lib/cmake/MultiAgentSolver
)
