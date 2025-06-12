cmake_minimum_required(VERSION 3.15)
project(MultiAgentSolver VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(BUILD_EXAMPLES "Build example executables" ON)
option(BUILD_SHARED_LIBS "Build as shared library" OFF)

# Dependencies
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(OpenMP REQUIRED)
find_package(osqp REQUIRED)
find_package(OsqpEigen REQUIRED)

# Interface library
add_library(MultiAgentSolver INTERFACE)

target_include_directories(MultiAgentSolver INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(MultiAgentSolver INTERFACE
  Eigen3::Eigen
  OsqpEigen::OsqpEigen
  osqp::osqp
  OpenMP::OpenMP_CXX
)

# Optional examples
if(BUILD_EXAMPLES)
  file(GLOB EXAMPLE_SOURCES CONFIGURE_DEPENDS examples/*.cpp)
  foreach(source_file IN LISTS EXAMPLE_SOURCES)
    get_filename_component(example_name ${source_file} NAME_WE)
    add_executable(${example_name} ${source_file})
    target_link_libraries(${example_name} PRIVATE MultiAgentSolver)
  endforeach()
endif()

# Install headers
install(DIRECTORY include/ DESTINATION include)

# Install target
install(TARGETS MultiAgentSolver EXPORT MultiAgentSolverTargets)

# Export targets
install(EXPORT MultiAgentSolverTargets
  FILE MultiAgentSolverTargets.cmake
  NAMESPACE MultiAgentSolver::
  DESTINATION lib/cmake/MultiAgentSolver
)

# Package config
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/MultiAgentSolverConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/MultiAgentSolverConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/MultiAgentSolverConfig.cmake"
  INSTALL_DESTINATION lib/cmake/MultiAgentSolver
)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/MultiAgentSolverConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/MultiAgentSolverConfigVersion.cmake"
  DESTINATION lib/cmake/MultiAgentSolver
)
