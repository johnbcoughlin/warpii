#pragma once

#include <deal.II/base/utilities.h>
#include <deal.II/base/vectorization.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <functional>

#include "../function_eval.h"
#include "bc_helper.h"
#include "dg_discretization.h"
#include "euler.h"
#include "species.h"

using namespace dealii;

namespace warpii {
namespace five_moment {

template <int dim>
class FluidFluxDGOperator {
   public:
    FluidFluxDGOperator(
        std::shared_ptr<FiveMomentDGDiscretization<dim>> discretization,
        double gas_gamma, std::vector<std::shared_ptr<Species<dim>>> species)
        : discretization(discretization),
          gas_gamma(gas_gamma),
          n_species(species.size()),
          species(species) {}

    void perform_forward_euler_step(
        LinearAlgebra::distributed::Vector<double> &dst,
        const LinearAlgebra::distributed::Vector<double> &u,
        std::vector<LinearAlgebra::distributed::Vector<double>> &sol_registers,
        const double dt, const double t, const double alpha = 1.0,
        const double beta = 0.0);

   private:
  void local_apply_inverse_mass_matrix(
      const MatrixFree<dim, double> &data,
      LinearAlgebra::distributed::Vector<double> &dst,
      const LinearAlgebra::distributed::Vector<double> &src,
      const std::pair<unsigned int, unsigned int> &cell_range) const;

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

    void local_apply_positivity_limiter(
        const MatrixFree<dim, double> &,
        LinearAlgebra::distributed::Vector<double> &dst,
        const LinearAlgebra::distributed::Vector<double> &src,
        const std::pair<unsigned int, unsigned int> &cell_range) const;

    std::shared_ptr<FiveMomentDGDiscretization<dim>> discretization;
    double gas_gamma;
    unsigned int n_species;
    std::vector<std::shared_ptr<Species<dim>>> species;
};

template <int dim>
void FluidFluxDGOperator<dim>::perform_forward_euler_step(
    LinearAlgebra::distributed::Vector<double> &dst,
    const LinearAlgebra::distributed::Vector<double> &u,
    std::vector<LinearAlgebra::distributed::Vector<double>> &sol_registers,
    const double dt, const double t, const double alpha, const double beta) {
    auto Mdudt_register = sol_registers.at(0);
    auto dudt_register = sol_registers.at(1);
    auto sol_before_limiting = sol_registers.at(2);

    {
        for (auto sp : species) {
            for (auto &i : sp->bc_map.inflow_boundaries()) {
                i.second->set_time(t);
            }
        }

        discretization->mf.loop(
            &FluidFluxDGOperator<dim>::local_apply_cell,
            &FluidFluxDGOperator<dim>::local_apply_face,
            &FluidFluxDGOperator<dim>::local_apply_boundary_face, this,
            Mdudt_register, u, true,
            MatrixFree<dim, double>::DataAccessOnFaces::values,
            MatrixFree<dim, double>::DataAccessOnFaces::values);
    }

    {
        discretization->mf.cell_loop(
            &FluidFluxDGOperator<dim>::local_apply_inverse_mass_matrix,
            this, dudt_register, Mdudt_register,
            std::function<void(const unsigned int, const unsigned int)>(),
            [&](const unsigned int start_range, const unsigned int end_range) {
                /* DEAL_II_OPENMP_SIMD_PRAGMA */
                for (unsigned int i = start_range; i < end_range; ++i) {
                    const double dudt_i = dudt_register.local_element(i);
                    const double dst_i = dst.local_element(i);
                    const double u_i = u.local_element(i);
                    sol_before_limiting.local_element(i) =
                        beta * dst_i + alpha * (u_i + dt * dudt_i);
                }
            });
    }

    {
        //discretization->mf.cell_loop(
            //&FluidFluxDGOperator<dim>::local_apply_positivity_limiter, this,
            //dst, sol_before_limiting);
        dst.sadd(0.0, 1.0, sol_before_limiting);
    }
}

template <int dim>
void FluidFluxDGOperator<dim>::
    local_apply_inverse_mass_matrix(
        const MatrixFree<dim, double> &mf,
        LinearAlgebra::distributed::Vector<double> &dst,
        const LinearAlgebra::distributed::Vector<double> &src,
        const std::pair<unsigned int, unsigned int> &cell_range) const {
  FEEvaluation<dim, -1, 0, dim + 2, double> phi(mf, 0, 1);
  MatrixFreeOperators::CellwiseInverseMassMatrix<dim, -1, dim + 2, double> inverse(phi);

  for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
    phi.reinit(cell);
    phi.read_dof_values(src);

    inverse.apply(phi.begin_dof_values(), phi.begin_dof_values());

    phi.set_dof_values(dst);
  }
}

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
        EulerBCMap<dim> &bc_map = species.at(species_index)->bc_map;
        unsigned int first_component = species_index * (dim + 2);
        FEFaceEvaluation<dim, -1, 0, dim + 2, double> phi(mf, true, 0, 0,
                                                          first_component);

        for (unsigned int face = face_range.first; face < face_range.second;
             ++face) {
            phi.reinit(face);
            phi.gather_evaluate(src, EvaluationFlags::values);

            const auto boundary_id = mf.get_boundary_id(face);

            for (const unsigned int q : phi.quadrature_point_indices()) {
                const Tensor<1, dim + 2, VectorizedArray<double>> w_m =
                    phi.get_value(q);
                const Tensor<1, 1, VectorizedArray<double>> normal =
                    phi.normal_vector(q);

                auto rho_u_dot_n = w_m[1] * normal[0];
                for (unsigned int d = 1; d < dim; d++) {
                    rho_u_dot_n += w_m[1 + d] * normal[d];
                }

                bool at_outflow = false;
                Tensor<1, dim + 2, VectorizedArray<double>> w_p;
                if (bc_map.is_inflow(boundary_id)) {
                    w_p = evaluate_function<dim, double>(
                        *bc_map.get_inflow(boundary_id),
                        phi.quadrature_point(q));
                } else if (bc_map.is_subsonic_outflow(boundary_id)) {
                    w_p = w_m;
                    w_p[dim + 1] = evaluate_function<dim, double>(
                        *bc_map.get_subsonic_outflow_energy(boundary_id),
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
                    for (unsigned int v = 0;
                         v < VectorizedArray<double>::size(); ++v) {
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

template <int dim>
void FluidFluxDGOperator<dim>::local_apply_positivity_limiter(
    const MatrixFree<dim, double> &mf,
    LinearAlgebra::distributed::Vector<double> &dst,
    const LinearAlgebra::distributed::Vector<double> &src,
    const std::pair<unsigned int, unsigned int> &cell_range) const {
    using VA = VectorizedArray<double>;

    // Used only for the area calculation
    FEEvaluation<dim, -1, 0, 1, double> phi_scalar(mf, 0, 1);
    // This should be constructed from the quadrature rule used for the
    // positivity-preserving step.
    // TODO: programmatically determine the number of quadrature points
    FEEvaluation<dim, -1, 0, dim + 2, double> phi(mf, 0, 1);
    MatrixFreeOperators::CellwiseInverseMassMatrix<dim, -1, dim+2, double> inverse(
        phi);

    for (unsigned int cell = cell_range.first; cell < cell_range.second;
         ++cell) {
        phi_scalar.reinit(cell);
        phi.reinit(cell);

        phi.gather_evaluate(src, EvaluationFlags::values);
        VA rho_min = VA(std::numeric_limits<double>::infinity());

        for (const unsigned int q : phi_scalar.quadrature_point_indices()) {
            phi_scalar.submit_value(VA(1.0), q);
        }
        auto area = phi_scalar.integrate_value();
        for (const unsigned int q : phi.quadrature_point_indices()) {
            auto v = phi.get_value(q);
            rho_min = std::min(v[0], rho_min);
            phi.submit_value(v, q);
        }
        auto cell_avg = phi.integrate_value() / area;

        auto rho_bar = cell_avg[0];
        auto p_bar = euler_pressure<dim>(cell_avg, gas_gamma);

        for (unsigned int v = 0; v < VA::size(); ++v) {
            if (rho_bar[v] <= 0.0) {
                AssertThrow(false,
                            ExcMessage("Cell average density was negative"));
            }
            if (p_bar[v] <= 0.0) {
                AssertThrow(false,
                            ExcMessage("Cell average pressure was negative"));
            }
        }

        /*
         * Theta_rho calculation
         */
        auto num_rho =
            std::max(rho_bar - (1e-12 * std::max(rho_bar, VA(1.0))), VA(0.0));
        auto denom_rho = rho_bar - rho_min;
        for (unsigned int v = 0; v < VA::size(); ++v) {
            denom_rho[v] = denom_rho[v] <= 0.0 ? 1.0 : denom_rho[v];
        }
        auto theta_rho = std::min(VA(1.0), num_rho / denom_rho);

        /*
         * Theta E calculation
         *
         * First we calculate the min energy after the theta_rho scaling
         */
        phi.gather_evaluate(src, EvaluationFlags::values);
        VA p_min = VA(std::numeric_limits<double>::infinity());
        for (const unsigned int q : phi.quadrature_point_indices()) {
            auto v = phi.get_value(q);
            v[0] = theta_rho * (v[0] - rho_bar) + rho_bar;
            p_min = std::min(euler_pressure<dim>(v, gas_gamma), p_min);
        }
        auto num_E = p_bar - (1e-12 * std::max(p_bar, VA(1.0)));
        auto denom_E = p_bar - p_min;
        for (unsigned int v = 0; v < VA::size(); ++v) {
            denom_E[v] = denom_E[v] <= 0.0 ? 1.0 : denom_E[v];
        }
        auto theta_E = std::min(VA(1.0), num_E / denom_E);

        for (unsigned int v = 0; v < VA::size(); ++v) {
            auto theta_rho_v = theta_rho[v];
            auto theta_E_v = theta_E[v];
            if (theta_rho_v != 1.0 || theta_E_v != 1.0) {
                // std::cout << "theta_rho: " << theta_rho_v << "; theta_E: " <<
                // theta_E_v << std::endl; std::cout << "theta_rho: " <<
                // theta_rho_v <<
                // "; denom_rho: " << denom_rho[v] << std::endl; std::cout <<
                // "min_rho: " << rho_min[v] << "; rho_bar: " << cell_avg[0][v]
                // << std::endl;
            }
        }

        // Finally, scale the quadrature point values by theta_rho and theta_E.
        phi.gather_evaluate(src, EvaluationFlags::values);
        for (const unsigned int q : phi.quadrature_point_indices()) {
            auto v = phi.get_value(q);
            // std::cout << "v: " << v << std::endl;
            auto rho = theta_rho * (v[0] - rho_bar) + rho_bar;
            rho = theta_E * (rho - rho_bar) + rho_bar;
            v[0] = rho;
            for (unsigned int c = 1; c < dim + 2; c++) {
                v[c] = theta_E * (v[c] - cell_avg[c]) + cell_avg[c];
            }
            auto pressure = euler_pressure<dim>(v, gas_gamma);
            for (unsigned int vec_i = 0; vec_i < VA::size(); ++vec_i) {
                // AssertThrow(v[dim+2][vec_i] > 1e-12, ExcMessage("Submitting
                // negative density to quad point"));
                if (pressure[vec_i] <= 1e-12) {
                    std::cout << "problem with: " << vec_i << std::endl;
                    std::cout << "cell avg: " << cell_avg << std::endl;
                    std::cout << "area: " << area << std::endl;
                    std::cout << "p bar: " << p_bar << std::endl;
                    std::cout << "p min: " << p_min << std::endl;
                    std::cout << "theta rho: " << theta_rho << std::endl;
                    std::cout << "theta rho: " << num_rho << std::endl;
                    std::cout << "theta rho: " << denom_rho << std::endl;
                    std::cout << "rho min: " << rho_min << std::endl;
                    std::cout << "theta E: " << theta_E << std::endl;
                    std::cout << "Submitting value: " << v << std::endl;
                }
                AssertThrow(
                    rho[vec_i] > 1e-12,
                    ExcMessage("Submitting negative density to quad point"));
                AssertThrow(
                    pressure[vec_i] > 1e-12,
                    ExcMessage("Submitting negative pressure to quad point"));
            }
            // std::cout << "v_submitted: " << v << std::endl;
            //  This overwrites the value previously submitted.
            //  See fe_evaluation.h:4995
            phi.submit_value(v, q);
        }
        phi.integrate(EvaluationFlags::values);
        inverse.apply(phi.begin_dof_values(), phi.begin_dof_values());
        phi.set_dof_values(dst);
    }
}

}  // namespace five_moment
}  // namespace warpii
