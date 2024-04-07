cmake_minimum_required(VERSION 3.21)
project(WarpII)

set(CMAKE_CXX_STANDARD 17)

if(DEFINED DEALII_INSTALL_DIR)
    list(INSERT CMAKE_PREFIX_PATH 0 ${DEALII_INSTALL_DIR})
endif()
find_package(deal.II $ENV{DEALII_VERSION} REQUIRED)

set(DEAL_II_WITH_MPI ON)

find_package(MPI REQUIRED)

deal_ii_initialize_cached_variables()

add_subdirectory(src)
add_subdirectory(codes)
