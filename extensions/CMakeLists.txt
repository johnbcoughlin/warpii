add_executable(dummy_extension dummy_extension_main.cc)
deal_ii_setup_target(dummy_extension)

target_link_libraries(dummy_extension libwarpii)
message("${CMAKE_SOURCE_DIR}")
target_include_directories(dummy_extension PRIVATE "${CMAKE_SOURCE_DIR}")

get_target_property(EXT_LINK_LIBRARIES dummy_extension LINK_LIBRARIES)

set(LIBWARPII_DIR "${CMAKE_BINARY_DIR}/src")

message("MPI DIR: ${MPI_CXX_LIBRARIES}")
list(GET MPI_CXX_LIBRARIES 0 MPI_CXX_LIB)
get_filename_component(MPI_DIR "${MPI_CXX_LIB}" DIRECTORY)
message("MPI DIR: ${MPI_DIR}")

get_target_property(DEAL_II_DEBUG_LIB ${DEAL_II_TARGET_DEBUG} LOCATION)
get_target_property(DEAL_II_RELEASE_LIB ${DEAL_II_TARGET_RELEASE} LOCATION)

set(EXT_LINK_TXT "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/dummy_extension.dir/link.txt")

add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/Makefile.example"
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/${WARPII_EXTENSION_MAKEFILE_GENERATOR} 
            ${CMAKE_BINARY_DIR}/compile_commands.json
            ${CMAKE_CXX_COMPILER}
            ${DEAL_II_DEBUG_LIB}
            ${DEAL_II_RELEASE_LIB}
            ${MPI_DIR}
            ${LIBWARPII_DIR}
            ${EXT_LINK_TXT}
    DEPENDS dummy_extension "${CMAKE_CURRENT_SOURCE_DIR}/${WARPII_EXTENSION_MAKEFILE_GENERATOR}")

add_custom_target(generate-example-makefile
    ALL
    DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/Makefile.example")
