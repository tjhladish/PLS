cmake_minimum_required( VERSION 3.24.2 )

project(PLS VERSION 2.0.0)

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
  )
  FetchContent_MakeAvailable(Eigen3)
endif(NOT Eigen3_FOUND)

if (DEFINED ${CMAKE_PROJECT_NAME}_SHARED_LIBS)
    set(BUILD_SHARED_LIBS "${${CMAKE_PROJECT_NAME}_SHARED_LIBS}")
endif ()

add_library(pls ${CMAKE_CURRENT_SOURCE_DIR}/src/pls.cpp)
add_library(${CMAKE_PROJECT_NAME}::pls ALIAS pls)

set_target_properties(pls PROPERTIES
                      VERSION ${${CMAKE_PROJECT_NAME}_VERSION}
                      SOVERSION ${${CMAKE_PROJECT_NAME}_VERSION_MAJOR})

target_include_directories(pls PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>")

target_compile_features(pls PUBLIC cxx_std_17)

target_compile_definitions(
    pls PUBLIC "$<$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>:PLS_STATIC_DEFINE>")
target_include_directories(
    pls PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>")

target_link_libraries(pls PUBLIC Eigen3::Eigen)

add_executable( PLS src/main.cpp )
target_link_libraries( PLS pls )

install(TARGETS PLS pls
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)
