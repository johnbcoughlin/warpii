add_library(libwarpii
    warpii.cc
    input_file.cc
    timestepper.cc
    grid.cc
    five_moment/five_moment.cc
    )
deal_ii_setup_target(libwarpii)

