cmake_minimum_required( VERSION 3.24.2 )

project( PLS )

add_library( pls STATIC src/pls.cpp )
target_include_directories(pls PUBLIC include lib/eigen)

add_executable( PLS src/main.cpp )
target_link_libraries( PLS pls )
