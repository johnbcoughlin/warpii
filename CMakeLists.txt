cmake_minimum_required(VERSION 3.21)
project(WarpII)

set(CMAKE_CXX_STANDARD 17)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")

if(DEFINED CACHE{WARPII_DOC_BUILD})
    include("warpii-doc")
endif()

if (WARPII_ONLY_DOCS)
    return()
endif()


####
####  Set up dependencies
####
find_package(MPI REQUIRED)

if(DEFINED DEALII_INSTALL_DIR)
    list(INSERT CMAKE_PREFIX_PATH 0 ${DEALII_INSTALL_DIR})
endif()
find_package(deal.II $ENV{DEALII_VERSION} REQUIRED)
set(DEAL_II_WITH_MPI ON)
deal_ii_initialize_cached_variables()

####
####  Our own source tree and main executable
####
add_subdirectory(src)

add_executable(warpii main.cc)
target_link_libraries(warpii libwarpii)
deal_ii_setup_target(warpii)

add_subdirectory(test)

