cmake_minimum_required(VERSION 3.15)
project(MultiAgentSolver VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(BUILD_EXAMPLES "Build example executables" ON)
option(BUILD_SHARED_LIBS "Build as shared library" OFF)
include(CTest)

# Dependencies
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(OpenMP REQUIRED)
# Optional OSQP dependencies
find_package(osqp QUIET)
find_package(OsqpEigen QUIET)
find_package(OpenMP REQUIRED)

# Interface library
add_library(MultiAgentSolver INTERFACE)

# Better stacks & symbols for perf in RelWithDebInfo
target_compile_options(MultiAgentSolver INTERFACE
  $<$<CONFIG:RelWithDebInfo>:-g -fno-omit-frame-pointer -fno-optimize-sibling-calls>
)
target_link_options(MultiAgentSolver INTERFACE
  $<$<CONFIG:RelWithDebInfo>:-Wl,--build-id>
)

target_compile_options(MultiAgentSolver INTERFACE ${OpenMP_CXX_FLAGS})
target_link_libraries(MultiAgentSolver  INTERFACE ${OSQP_LIBRARIES} ${OpenMP_CXX_LIBRARIES})

target_include_directories(MultiAgentSolver INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(MultiAgentSolver INTERFACE
  Eigen3::Eigen
  OpenMP::OpenMP_CXX
)
if(osqp_FOUND AND OsqpEigen_FOUND)
  target_link_libraries(MultiAgentSolver INTERFACE OsqpEigen::OsqpEigen osqp::osqp)
  target_compile_definitions(MultiAgentSolver INTERFACE MAS_HAVE_OSQP)
endif()

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

if(BUILD_TESTING)
  include(FetchContent)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
  )
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  enable_testing()
  file(GLOB TEST_SOURCES CONFIGURE_DEPENDS tests/*.cpp)
  add_executable(multi_agent_solver_tests ${TEST_SOURCES})
  target_link_libraries(multi_agent_solver_tests PRIVATE MultiAgentSolver GTest::gtest_main)

  include(GoogleTest)
  gtest_discover_tests(multi_agent_solver_tests)
endif()
