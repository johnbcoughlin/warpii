#pragma once

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include "bc_helper.h"
#include "euler.h"
#include "../function_eval.h"

using namespace dealii;

namespace warpii {
namespace five_moment {

template <int dim>
class FluidFluxDGOperator {
   public:
    void local_apply_cell(
        const MatrixFree<dim, double> &mf,
        LinearAlgebra::distributed::Vector<double> &dst,
        const LinearAlgebra::distributed::Vector<double> &src,
        const std::pair<unsigned int, unsigned int> &cell_range) const;

    void local_apply_face(
        const MatrixFree<dim, double> &mf,
        LinearAlgebra::distributed::Vector<double> &dst,
        const LinearAlgebra::distributed::Vector<double> &src,
        const std::pair<unsigned int, unsigned int> &face_range) const;

    void local_apply_boundary_face(
        const MatrixFree<dim, double> &,
        LinearAlgebra::distributed::Vector<double> &dst,
        const LinearAlgebra::distributed::Vector<double> &src,
        const std::pair<unsigned int, unsigned int> &face_range) const;

   private:
    double gas_gamma;
    unsigned int n_species;
    std::vector<EulerBCMap<dim>> bc_maps;
};

template <int dim>
void FluidFluxDGOperator<dim>::local_apply_cell(
    const MatrixFree<dim, double> &mf,
    LinearAlgebra::distributed::Vector<double> &dst,
    const LinearAlgebra::distributed::Vector<double> &src,
    const std::pair<unsigned int, unsigned int> &cell_range) const {
    for (unsigned int species_index = 0; species_index < n_species;
         species_index++) {
        unsigned int first_component = species_index * (dim + 2);
        FEEvaluation<dim, -1, 0, dim + 2, double> phi(mf, 0, 0,
                                                      first_component);

        for (unsigned int cell = cell_range.first; cell < cell_range.second;
             ++cell) {
            phi.reinit(cell);
            phi.gather_evaluate(src, EvaluationFlags::values);

            for (const unsigned int q : phi.quadrature_point_indices()) {
                const auto w_q = phi.get_value(q);
                phi.submit_gradient(euler_flux<dim>(w_q, gas_gamma), q);
            }

            phi.integrate_scatter(EvaluationFlags::gradients, dst);
        }
    }
}

template <int dim>
void FluidFluxDGOperator<dim>::local_apply_face(
    const MatrixFree<dim, double> &mf,
    LinearAlgebra::distributed::Vector<double> &dst,
    const LinearAlgebra::distributed::Vector<double> &src,
    const std::pair<unsigned int, unsigned int> &face_range) const {
    for (unsigned int species_index = 0; species_index < n_species;
         species_index++) {
        unsigned int first_component = species_index * (dim + 2);
        FEFaceEvaluation<dim, -1, 0, dim + 2, double> phi_m(mf, true, 0, 0,
                                                            first_component);
        FEFaceEvaluation<dim, -1, 0, dim + 2, double> phi_p(mf, false, 0, 0,
                                                            first_component);

        for (unsigned int face = face_range.first; face < face_range.second;
             ++face) {
            phi_p.reinit(face);
            phi_p.gather_evaluate(src, EvaluationFlags::values);

            phi_m.reinit(face);
            phi_m.gather_evaluate(src, EvaluationFlags::values);

            for (const unsigned int q : phi_m.quadrature_point_indices()) {
                const auto numerical_flux = euler_numerical_flux<dim>(
                    phi_m.get_value(q), phi_p.get_value(q),
                    phi_m.normal_vector(q), gas_gamma);
                phi_m.submit_value(-numerical_flux, q);
                phi_p.submit_value(numerical_flux, q);
            }

            phi_p.integrate_scatter(EvaluationFlags::values, dst);
            phi_m.integrate_scatter(EvaluationFlags::values, dst);
        }
    }
}

template <int dim>
void FluidFluxDGOperator<dim>::local_apply_boundary_face(
    const MatrixFree<dim> &mf, LinearAlgebra::distributed::Vector<double> &dst,
    const LinearAlgebra::distributed::Vector<double> &src,
    const std::pair<unsigned int, unsigned int> &face_range) const {
    for (unsigned int species_index = 0; species_index < n_species;
         species_index++) {
        EulerBCMap<dim> &bc_map = bc_maps.at(species_index);
        unsigned int first_component = species_index * (dim + 2);
        FEFaceEvaluation<dim, -1, 0, dim + 2, double> phi(mf, true, 0, 0,
                                                          first_component);

        for (unsigned int face = face_range.first; face < face_range.second;
             ++face) {
            phi.reinit(face);
            phi.gather_evaluate(src, EvaluationFlags::values);

            const auto boundary_id = mf.get_boundary_id(face);

            for (const unsigned int q : phi.quadrature_point_indices()) {
                const auto w_m = phi.get_value(q);
                const auto normal = phi.normal_vector(q);

                auto rho_u_dot_n = w_m[1] * normal[0];
                for (unsigned int d = 1; d < dim; d++) {
                    rho_u_dot_n += w_m[1 + d] * normal[d];
                }

                bool at_outflow = false;
                Tensor<1, dim + 2, VectorizedArray<double>> w_p;
                if (bc_map.is_inflow(boundary_id)) {
                    w_p = evaluate_function<dim, double>(
                        bc_map.get_inflow(boundary_id),
                        phi.quadrature_point(q));
                } else if (bc_map.is_subsonic_outflow(boundary_id)) {
                    w_p = w_m;
                    w_p[dim + 1] = evaluate_function<dim, double>(
                        bc_map.get_subsonic_outflow_energy(boundary_id),
                        phi.quadrature_point(q), dim + 1);
                    at_outflow = true;
                } else if (bc_map.is_supersonic_outflow(boundary_id)) {
                    w_p = w_m;
                    at_outflow = true;
                } else if (bc_map.is_wall(boundary_id)) {
                    // Copy out density
                    w_p[0] = w_m[0];
                    for (unsigned int d = 0; d < dim; d++) {
                        w_p[d + 1] = w_m[d + 1] - 2.0 * rho_u_dot_n * normal[d];
                    }
                    w_p[dim + 1] = w_m[dim + 1];
                } else {
                    AssertThrow(
                        false,
                        ExcMessage("Unknown boundary id, did you set a "
                                   "boundary condition "
                                   "for this part of the domain boundary?"));
                }

                auto flux =
                    euler_numerical_flux<dim>(w_m, w_p, normal, gas_gamma);

                if (at_outflow) {
                    for (unsigned int v = 0; v < VectorizedArray<double>::size(); ++v) {
                        if (rho_u_dot_n[v] < -1e-12) {
                            for (unsigned int d = 0; d < 1; ++d) {
                                flux[d + 1][v] = 0.;
                            }
                        }
                    }
                }

                phi.submit_value(-flux, q);
            }
            phi.integrate_scatter(EvaluationFlags::values, dst);
        }
    }
}

}  // namespace five_moment
}  // namespace warpii
