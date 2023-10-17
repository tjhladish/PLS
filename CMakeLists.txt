cmake_minimum_required( VERSION 3.16 )

project(PLS VERSION 2.0.1 LANGUAGES CXX)

#include(CMakePackageConfigHelpers)

# configure_package_config_file(
#   "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
#   INSTALL_DESTINATION
#   ${CMAKE_INSTALL_LIBDIR}/cmake/PLS-${PROJECT_VERSION}
# )

# TODO need to figure out the right way to ask for MPREAL_SUPPORT
# would be propogated from parent project, presumably

# TODO figure out if this is the right way to do this
# intend: if new enough Eigen3 is found, use it;
# otherwise, install from submoduled source
find_package(Eigen3 3.4.90)

if (NOT Eigen3_FOUND)
  message(WARNING "Eigen3 3.4.90+ not found; installing from source...")
  include(FetchContent)
  FetchContent_Declare(
    Eigen3
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib/eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  )
  FetchContent_MakeAvailable(Eigen3)
else()
  message(STATUS "Eigen3 found: ${Eigen3_VERSION}")
endif()

if (DEFINED ${CMAKE_PROJECT_NAME}_SHARED_LIBS)
    set(BUILD_SHARED_LIBS "${${CMAKE_PROJECT_NAME}_SHARED_LIBS}")
endif ()

# include(GenerateExportHeader)

add_library(PLS ${CMAKE_CURRENT_SOURCE_DIR}/src/pls.cpp)

set_target_properties(PLS PROPERTIES
                      VERSION ${${CMAKE_PROJECT_NAME}_VERSION}
                      SOVERSION ${${CMAKE_PROJECT_NAME}_VERSION_MAJOR})

# generate_export_header(pls)

target_compile_features(PLS PUBLIC cxx_std_17)

target_compile_definitions(
  PLS PUBLIC "$<$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>:PLS_STATIC_DEFINE>"
)

target_include_directories(
  PLS PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>"
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include>"
)

target_link_libraries(PLS PUBLIC Eigen3::Eigen)

add_executable(PLS-bin src/main.cpp)
target_link_libraries(PLS-bin PLS)
set_target_properties(PLS-bin PROPERTIES OUTPUT_NAME PLS)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "PLSConfigVersion.cmake"
  VERSION ${PLS_VERSION}
  COMPATIBILITY SameMajorVersion
)
install(FILES "PLSConfig.cmake" "${CMAKE_CURRENT_BINARY_DIR}/PLSConfigVersion.cmake"
  DESTINATION lib/cmake/PLS
)

install(TARGETS PLS EXPORT PLSTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT PLSTargets
  FILE PLSTargets.cmake
  NAMESPACE PLS::
  DESTINATION lib/cmake/PLS
)

install(TARGETS PLS-bin
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)