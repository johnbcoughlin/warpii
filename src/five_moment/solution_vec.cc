#include "solution_vec.h"

namespace warpii {
    namespace five_moment {
        void FiveMSolutionVec::reinit(const FiveMSolutionVec& other) {
            mesh_sol.reinit(other.mesh_sol);
            boundary_integrated_fluxes.reinit(other.boundary_integrated_fluxes);
        }

        void FiveMBoundaryIntegratedFluxesVector::sadd(const double s, const double a, const FiveMBoundaryIntegratedFluxesVector& V) {
            data.sadd(s, a, V.data);
        }

        void FiveMBoundaryIntegratedFluxesVector::reinit(unsigned int n_boundaries, unsigned int) {
            data.reinit(n_boundaries * 5);
        }

        void FiveMBoundaryIntegratedFluxesVector::reinit(const FiveMBoundaryIntegratedFluxesVector& other) {
            data.reinit(other.data);
        }

        bool FiveMBoundaryIntegratedFluxesVector::is_empty() {
            return data.size() == 0;
        }
    }
}
