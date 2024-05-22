#pragma once

#include <deal.II/lac/vector.h>
#include <deal.II/lac/la_parallel_vector.h>

using namespace dealii;

namespace warpii {
    namespace five_moment {
        struct FiveMSolutionVec {
            LinearAlgebra::distributed::Vector<double> mesh_sol;
            Vector<double> boundary_integrated_fluxes;
        };

        void reinit_solution_vec(FiveMSolutionVec& a, FiveMSolutionVec& b);
    }
}
