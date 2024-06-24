#pragma once

#include <deal.II/lac/vector.h>
#include <deal.II/lac/la_parallel_vector.h>

using namespace dealii;

namespace warpii {
    namespace five_moment {
        class FiveMBoundaryIntegratedFluxesVector {
            public:
                template <int dim>
                void add(unsigned int boundary_id, Tensor<1, 5, double> flux);

                void sadd(const double s, const double a, const FiveMBoundaryIntegratedFluxesVector& V);

                void reinit(unsigned int n_boundaries, unsigned int n_dims);

                void reinit(const FiveMBoundaryIntegratedFluxesVector& other);

                template <int dim>
                Tensor<1, 5, double> at_boundary(unsigned int boundary_id);

                bool is_empty();

                Vector<double> data;
        };

        template <int dim>
        void FiveMBoundaryIntegratedFluxesVector::add(unsigned int boundary_id, Tensor<1, 5, double> flux) {
            for (unsigned int comp = 0; comp < 5; comp++) {
                data[boundary_id * (5) + comp] += flux[comp];
            }
        }

        template <int dim>
        Tensor<1, 5, double> FiveMBoundaryIntegratedFluxesVector::at_boundary(unsigned int boundary_id) {
            Tensor<1, 5, double> result;
            for (unsigned int comp = 0; comp < 5; comp++) {
                result[comp] = data[boundary_id * (5) + comp];
            }
            return result;
        }

        class FiveMSolutionVec {
            public:
                LinearAlgebra::distributed::Vector<double> mesh_sol;
                FiveMBoundaryIntegratedFluxesVector boundary_integrated_fluxes;

            void reinit(const FiveMSolutionVec& other);
        };
    }
}
