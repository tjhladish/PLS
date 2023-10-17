cmake_minimum_required( VERSION 3.16 )

project(PLS VERSION 2.0.0)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# TODO need to figure out the right way to ask for MPREAL_SUPPORT
# would be propogated from parent project, presumably

# TODO figure out if this is the right way to do this
# intend: if new enough Eigen3 is found, use it;
# otherwise, install from submoduled source
find_package(Eigen3 3.4.90 NO_MODULE)

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

add_library(pls ${CMAKE_CURRENT_SOURCE_DIR}/src/pls.cpp)
add_library(${CMAKE_PROJECT_NAME}::pls ALIAS pls)

set_target_properties(pls PROPERTIES
                      VERSION ${${CMAKE_PROJECT_NAME}_VERSION}
                      SOVERSION ${${CMAKE_PROJECT_NAME}_VERSION_MAJOR})

# generate_export_header(pls)

target_compile_features(pls PUBLIC cxx_std_17)

target_compile_definitions(
  pls PUBLIC "$<$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>:PLS_STATIC_DEFINE>"
)

target_include_directories(
  pls PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

target_link_libraries(pls PUBLIC Eigen3::Eigen)

add_executable( PLS src/main.cpp )
target_link_libraries( PLS pls )

install(TARGETS PLS pls
  EXPORT plsTargets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# if (${CMAKE_INSTALL_INCLUDEDIR})
#   message(STATUS "CMAKE_INSTALL_INCLUDEDIR: ${CMAKE_INSTALL_INCLUDEDIR}")
#   install(FILES pls.h types.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
# else()
#   message(STATUS "CMAKE_INSTALL_INCLUDEDIR not defined")
# endif()

install(EXPORT plsTargets
        FILE plsTargets.cmake
        NAMESPACE PLS::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/pls
)