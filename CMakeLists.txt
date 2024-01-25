cmake_minimum_required(VERSION 3.21)
project(WarpII)

set(CMAKE_CXX_STANDARD 17)
find_package(deal.II 9.5.0 REQUIRED HINTS $ENV{LD_LIBRARY_PATH})

set(DEAL_II_WITH_MPI ON)

find_package(MPI REQUIRED)

include_directories(/Users/jack/usr/include)
link_directories(/Users/jack/usr/lib)

deal_ii_initialize_cached_variables()

add_subdirectory(src)

add_executable(main src/main.cc)
deal_ii_setup_target(main)
