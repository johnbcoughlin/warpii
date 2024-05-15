find_package(Doxygen)

set(doxyfile "${CMAKE_CURRENT_BINARY_DIR}/doxyfile")
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/docs/doxyfile.in"
    ${doxyfile} @ONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/docs/doxylayout.in.xml"
    "${CMAKE_CURRENT_BINARY_DIR}/doxylayout.xml" COPYONLY)

set(docs_src_dir "${CMAKE_CURRENT_SOURCE_DIR}/docs")
set(docs_dest_dir "${CMAKE_CURRENT_BINARY_DIR}/docs")

add_custom_target(copy-docs
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${docs_src_dir} ${docs_dest_dir}
    COMMENT "Copying docs/ directory to build tree")

add_custom_target(doxygen
    COMMAND ${DOXYGEN_EXECUTABLE} ${doxyfile}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating documentation with Doxygen"
    VERBATIM)

add_dependencies(doxygen copy-docs)

