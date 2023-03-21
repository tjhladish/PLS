cmake_minimum_required(VERSION 3.19)

project(example)

enable_testing()

find_package(PLS 2 REQUIRED ${example_pls_components})

add_executable(main main.cpp)
target_link_libraries(main PRIVATE PLS::pls)

add_test(NAME random_is_42 COMMAND main)
set_tests_properties(random_is_42 PROPERTIES
                     PASS_REGULAR_EXPRESSION "is: 42"
                     ENVIRONMENT "PATH=$<TARGET_FILE_DIR:PLS::pls>")

