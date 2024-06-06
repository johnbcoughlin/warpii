add_test(
    NAME ExtensionTutorialTest
    COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/extension_tutorial_test.sh"
            "${CMAKE_SOURCE_DIR}/docs/extension_tutorial.md"
            "${CMAKE_CURRENT_SOURCE_DIR}/print_nth_code_block.sh"
            "${CMAKE_SOURCE_DIR}"
)
