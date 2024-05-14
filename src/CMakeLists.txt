add_library(libwarpii
    warpii.cc
    input_file.cc
    timestepper.cc
    grid.cc
    dof_utils.cc
    five_moment/five_moment.cc
    fpe.cc
    )
deal_ii_setup_target(libwarpii)

