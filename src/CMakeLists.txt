add_library(libwarpii
    warpii.cc
    input_file.cc
    timestepper.cc
    grid.cc
    dof_utils.cc
    five_moment/dg_discretization.cc
    five_moment/dg_solver.cc
    five_moment/five_moment.cc
    five_moment/solution_vec.cc
    five_moment/species.cc
    fpe.cc
    )
deal_ii_setup_target(libwarpii)

