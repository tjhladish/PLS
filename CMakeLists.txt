cmake_minimum_required( VERSION 3.24.2 )

project( PLS )

set( CMAKE_CXX_STANDARD 20 )
set( CMAKE_CXX_FLAGS "-Wall -Wpedantic -O2 -fPIC" )

include_directories( include lib/eigen )

add_executable( PLS src/main.cpp )

add_library( pls STATIC src/pls.cpp )
target_link_libraries( PLS pls )

find_package (Eigen3 3.4.90 REQUIRED NO_MODULE)
target_link_libraries (PLS Eigen3::Eigen)
