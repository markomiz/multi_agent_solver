# Minimal Eigen3 finder for environments without the official CMake package.

find_path(
  EIGEN3_INCLUDE_DIR
  NAMES Eigen/Core
  PATH_SUFFIXES eigen3
)

if(NOT EIGEN3_INCLUDE_DIR)
  find_path(
    EIGEN3_INCLUDE_DIR
    NAMES Eigen/Core
  )
endif()

if(EIGEN3_INCLUDE_DIR)
  file(READ "${EIGEN3_INCLUDE_DIR}/Eigen/src/Core/util/Macros.h" _eigen_macros
    LIMIT 4096)
  string(REGEX MATCH "#define EIGEN_WORLD_VERSION ([0-9]+)" _world_match "${_eigen_macros}")
  string(REGEX MATCH "#define EIGEN_MAJOR_VERSION ([0-9]+)" _major_match "${_eigen_macros}")
  string(REGEX MATCH "#define EIGEN_MINOR_VERSION ([0-9]+)" _minor_match "${_eigen_macros}")
  if(_world_match AND _major_match AND _minor_match)
    string(REGEX REPLACE ".* ([0-9]+)" "\\1" EIGEN3_WORLD_VERSION "${_world_match}")
    string(REGEX REPLACE ".* ([0-9]+)" "\\1" EIGEN3_MAJOR_VERSION "${_major_match}")
    string(REGEX REPLACE ".* ([0-9]+)" "\\1" EIGEN3_MINOR_VERSION "${_minor_match}")
    set(EIGEN3_VERSION_STRING
      "${EIGEN3_WORLD_VERSION}.${EIGEN3_MAJOR_VERSION}.${EIGEN3_MINOR_VERSION}")
  endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Eigen3
  REQUIRED_VARS EIGEN3_INCLUDE_DIR
  VERSION_VAR EIGEN3_VERSION_STRING)

if(Eigen3_FOUND AND NOT TARGET Eigen3::Eigen)
  add_library(Eigen3::Eigen INTERFACE IMPORTED)
  set_target_properties(Eigen3::Eigen PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIR}")
endif()

mark_as_advanced(EIGEN3_INCLUDE_DIR)
