add_library(libwarpii
    warpii.cc
    input_file.cc
    grid.cc
    five_moment/five_moment.cc
    five_moment/dg_operator.cc
    )
deal_ii_setup_target(libwarpii)

