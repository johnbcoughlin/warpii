include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

set(TEST_SOURCES
    input_test.cc
    timestepper_test.cc
    euler_test.cc
    conservation_test.cc
    dof_utils_test.cc
    shock_capturing_fv_test.cc
    utilities_test.cc
    # Add more test files here as needed
)

foreach(TEST_SOURCE ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
    add_executable(${TEST_NAME} ${TEST_SOURCE}
        euler_test_helpers.cc)
    target_link_libraries(${TEST_NAME} GTest::gtest_main libwarpii)
    target_include_directories(${TEST_NAME} PRIVATE "${CMAKE_SOURCE_DIR}")
    deal_ii_setup_target(${TEST_NAME})
    gtest_discover_tests(${TEST_NAME})
endforeach()

add_test(
    NAME HelpTest
    COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/scripts/run-help.sh" "${CMAKE_BINARY_DIR}/warpii"
)
add_test(
    NAME FPETest
    COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/scripts/fpe-enabled-test.sh" "${CMAKE_BINARY_DIR}/warpii"
)
add_test(
    NAME CompileExtensionTest
    COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/scripts/compile-extension.sh" "${CMAKE_BINARY_DIR}/extensions/Makefile.example"
)
add_test(
    NAME Pragmas
    COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/scripts/headers-start-with-pragma-once.sh"
)
set_tests_properties(Pragmas PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
