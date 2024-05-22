#pragma once

#include <deal.II/lac/vector.h>
#include <deal.II/lac/la_parallel_vector.h>

using namespace dealii;

namespace warpii {
    namespace five_moment {
        class FiveMBoundaryIntegratedFluxesVector {
            public:
                template <int dim>
                void add(unsigned int boundary_id, Tensor<1, dim+2, double> flux);

                void sadd(const double s, const double a, const FiveMBoundaryIntegratedFluxesVector& V);

                void reinit(unsigned int n_boundaries, unsigned int n_dims);

                void reinit(const FiveMBoundaryIntegratedFluxesVector& other);

                Vector<double> data;
        };

        template <int dim>
        void FiveMBoundaryIntegratedFluxesVector::add(unsigned int boundary_id, Tensor<1, dim+2, double> flux) {
            for (unsigned int comp = 0; comp < dim+2; comp++) {
                data[boundary_id * (dim+2) + comp] += flux[comp];
            }
        }

        class FiveMSolutionVec {
            public:
                LinearAlgebra::distributed::Vector<double> mesh_sol;
                FiveMBoundaryIntegratedFluxesVector boundary_integrated_fluxes;

            void reinit(const FiveMSolutionVec& other);
        };
    }
}
