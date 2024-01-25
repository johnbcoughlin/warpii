add_library(libwarpii
        rk.cc
        Simulation.h
        DGOperator.h
)

deal_ii_setup_target(libwarpii)
