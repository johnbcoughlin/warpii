include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(
  input_test
  input_test.cc
)
target_link_libraries(
  input_test
  GTest::gtest_main
  libwarpii
)
target_include_directories(input_test PRIVATE "${CMAKE_SOURCE_DIR}")
deal_ii_setup_target(input_test)

include(GoogleTest)
gtest_discover_tests(input_test)

add_test(
    NAME HelpTest
    COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/scripts/run-help.sh" "${CMAKE_BINARY_DIR}/warpii"
)
add_test(
    NAME Pragmas
    COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/scripts/headers-start-with-pragma-once.sh"
)
set_tests_properties(Pragmas PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
