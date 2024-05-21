#include "solution_vec.h"

namespace warpii {
    namespace five_moment {
        void reinit_solution_vec(FiveMSolutionVec& a, FiveMSolutionVec& b) {
            a.mesh_sol.reinit(b.mesh_sol);
            a.nonmesh_sol.reinit(b.nonmesh_sol);
        }
    }
}
