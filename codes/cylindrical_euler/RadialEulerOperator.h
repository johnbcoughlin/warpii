#pragma once

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/time_stepping.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/vectorization.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/numerics/data_out.h>

#include <fstream>
#include <iomanip>
#include <iostream>

#include "euler/bc_helper.h"
#include "function_eval.h"
#include "radial_euler.h"

namespace CylindricalEuler {
using namespace dealii;

using Number = double;

template <int dim, int degree, int n_points_1d>
class RadialEulerOperator {
   public:
    RadialEulerOperator(TimerOutput &timer, double gamma)
        : timer(timer), gamma(gamma) {}

    void reinit(const Mapping<dim> &mapping,
                const DoFHandler<dim> &dof_handler);

    void set_inflow_boundary(const types::boundary_id boundary_id,
                             std::unique_ptr<Function<dim>> inflow_function);
    void set_subsonic_outflow_boundary(
        const types::boundary_id boundary_id,
        std::unique_ptr<Function<dim>> outflow_energy);
    void set_supersonic_outflow_boundary(const types::boundary_id boundary_id);
    void set_axial_boundary(const types::boundary_id boundary_id);
    void set_wall_boundary(const types::boundary_id boundary_id);

    void perform_stage(
        const Number cur_time, const Number factor_solution,
        const Number factor_ai,
        const LinearAlgebra::distributed::Vector<Number> &current_ri,
        LinearAlgebra::distributed::Vector<Number> &vec_ki,
        LinearAlgebra::distributed::Vector<Number> &solution,
        LinearAlgebra::distributed::Vector<Number> &next_ri) const;

    void initialize_vector(
        LinearAlgebra::distributed::Vector<Number> &vector) const;

    void project(const Function<dim> &function,
                 LinearAlgebra::distributed::Vector<Number> &solution) const;

    double compute_cell_transport_speed(
        const LinearAlgebra::distributed::Vector<Number> &solution) const;

    const EulerBCMap<dim>& bc_map() const {
        return _bc_map;
    }

    EulerBCMap<dim>& bc_map() {
        return _bc_map;
    }

   private:
    MatrixFree<dim, Number> data;
    TimerOutput &timer;
    double gamma;
    EulerBCMap<dim> _bc_map;

    void local_apply_inverse_mass_matrix(
        const MatrixFree<dim, Number> &data,
        LinearAlgebra::distributed::Vector<Number> &dst,
        const LinearAlgebra::distributed::Vector<Number> &src,
        const std::pair<unsigned int, unsigned int> &cell_range) const;

    void local_apply_cell(
        const MatrixFree<dim, Number> &data,
        LinearAlgebra::distributed::Vector<Number> &dst,
        const LinearAlgebra::distributed::Vector<Number> &src,
        const std::pair<unsigned int, unsigned int> &cell_range) const;

    void local_apply_face(
        const MatrixFree<dim, Number> &data,
        LinearAlgebra::distributed::Vector<Number> &dst,
        const LinearAlgebra::distributed::Vector<Number> &src,
        const std::pair<unsigned int, unsigned int> &face_range) const;

    void local_apply_boundary_face(
        const MatrixFree<dim, Number> &data,
        LinearAlgebra::distributed::Vector<Number> &dst,
        const LinearAlgebra::distributed::Vector<Number> &src,
        const std::pair<unsigned int, unsigned int> &face_range) const;
};

template <int dim, int degree, int n_points_1d>
void RadialEulerOperator<dim, degree, n_points_1d>::reinit(
    const Mapping<dim> &mapping, const DoFHandler<dim> &dof_handler) {
    const std::vector<const DoFHandler<dim> *> dof_handlers = {&dof_handler};
    const AffineConstraints<double> dummy;
    const std::vector<const AffineConstraints<double> *> constraints = {&dummy};
    const std::vector<Quadrature<1>> quadratures = {QGauss<1>(n_points_1d),
                                                    QGauss<1>(degree + 1)};

    typename MatrixFree<dim, Number>::AdditionalData additional_data;
    additional_data.mapping_update_flags =
        (update_gradients | update_JxW_values | update_quadrature_points |
         update_values);
    additional_data.mapping_update_flags_inner_faces =
        (update_JxW_values | update_quadrature_points | update_normal_vectors |
         update_values);
    additional_data.mapping_update_flags_boundary_faces =
        (update_JxW_values | update_quadrature_points | update_normal_vectors |
         update_values);
    additional_data.tasks_parallel_scheme =
        MatrixFree<dim, Number>::AdditionalData::none;

    data.reinit(mapping, dof_handlers, constraints, quadratures,
                additional_data);
}

template <int dim, int degree, int n_points_1d>
void RadialEulerOperator<dim, degree, n_points_1d>::initialize_vector(
    LinearAlgebra::distributed::Vector<Number> &vector) const {
    data.initialize_dof_vector(vector);
}

template <int dim, int degree, int n_points_1d>
void RadialEulerOperator<dim, degree, n_points_1d>::local_apply_cell(
    const MatrixFree<dim, Number> &,
    LinearAlgebra::distributed::Vector<Number> &dst,
    const LinearAlgebra::distributed::Vector<Number> &src,
    const std::pair<unsigned int, unsigned int> &cell_range) const {
    FEEvaluation<dim, degree, n_points_1d, dim + 2, Number> phi(data);

    for (unsigned int cell = cell_range.first; cell < cell_range.second;
         ++cell) {
        phi.reinit(cell);
        phi.gather_evaluate(src, EvaluationFlags::values);

        for (const unsigned int q : phi.quadrature_point_indices()) {
            const auto w_q = phi.get_value(q);
            phi.submit_gradient(
                radial_euler_flux<dim, VectorizedArray<Number>>(w_q, gamma), q);

            Tensor<1, dim + 2, VectorizedArray<Number>> source;
            const auto rp_q =
                radial_euler_pressure<dim, VectorizedArray<Number>>(w_q, gamma);
            const auto r = phi.quadrature_point(q)[0];
            source[1] = rp_q / r;
            // std::cout << "p: " << source[1] << ", rpq: " << rp_q <<
            // std::endl;
            phi.submit_value(source, q);
        }

        phi.integrate_scatter(
            EvaluationFlags::values | EvaluationFlags::gradients, dst);
    }
}

template <int dim, int degree, int n_points_1d>
void RadialEulerOperator<dim, degree, n_points_1d>::local_apply_face(
    const MatrixFree<dim, Number> &,
    LinearAlgebra::distributed::Vector<Number> &dst,
    const LinearAlgebra::distributed::Vector<Number> &src,
    const std::pair<unsigned int, unsigned int> &face_range) const {
    FEFaceEvaluation<dim, degree, n_points_1d, dim + 2, Number> phi_m(data,
                                                                      true);
    FEFaceEvaluation<dim, degree, n_points_1d, dim + 2, Number> phi_p(data,
                                                                      false);

    for (unsigned int face = face_range.first; face < face_range.second;
         ++face) {
        phi_p.reinit(face);
        phi_p.gather_evaluate(src, EvaluationFlags::values);

        phi_m.reinit(face);
        phi_m.gather_evaluate(src, EvaluationFlags::values);

        for (const unsigned int q : phi_m.quadrature_point_indices()) {
            const auto r = phi_m.quadrature_point(q);
            const auto numerical_flux = radial_euler_numerical_flux<dim>(
                phi_m.get_value(q), phi_p.get_value(q), phi_m.normal_vector(q),
                r, gamma);
            phi_m.submit_value(-numerical_flux, q);
            phi_p.submit_value(numerical_flux, q);
        }

        phi_p.integrate_scatter(EvaluationFlags::values, dst);
        phi_m.integrate_scatter(EvaluationFlags::values, dst);
    }
}

template <int dim, int degree, int n_points_1d>
void RadialEulerOperator<dim, degree, n_points_1d>::local_apply_boundary_face(
    const MatrixFree<dim, Number> &,
    LinearAlgebra::distributed::Vector<Number> &dst,
    const LinearAlgebra::distributed::Vector<Number> &src,
    const std::pair<unsigned int, unsigned int> &face_range) const {
    FEFaceEvaluation<dim, degree, n_points_1d, dim + 2, Number> phi(data, true);

    for (unsigned int face = face_range.first; face < face_range.second;
         ++face) {
        phi.reinit(face);
        phi.gather_evaluate(src, EvaluationFlags::values);

        for (const unsigned int q : phi.quadrature_point_indices()) {
            const auto w_m = phi.get_value(q);
            const auto normal = phi.normal_vector(q);

            auto r_rho_u_dot_n = w_m[1] * normal[0];
            for (unsigned int d = 1; d < dim; ++d) {
                r_rho_u_dot_n += w_m[1 + d] * normal[d];
            }

            bool at_outflow = false;

            Tensor<1, dim + 2, VectorizedArray<Number>> w_p;
            const auto boundary_id = data.get_boundary_id(face);
            if (_bc_map.is_inflow(boundary_id)) {
                w_p = evaluate_function<dim, Number, VectorizedArray<Number>>(*_bc_map.get_inflow(boundary_id),
                                        phi.quadrature_point(q));
            } else if (_bc_map.is_subsonic_outflow(boundary_id)) {
                w_p = w_m;
                w_p[dim + 1] = evaluate_function<dim, Number, VectorizedArray<Number>>(
                    *_bc_map.get_subsonic_outflow_energy(boundary_id),
                    phi.quadrature_point(q), dim + 1);
                at_outflow = true;
            } else if (_bc_map.is_supersonic_outflow(boundary_id)) {
                w_p = w_m;
                at_outflow = true;
            } else if (_bc_map.is_axial(boundary_id)) {
                // Copy out density
                w_p[0] = w_m[0];
                // Zero radial velocity via mirror principle
                w_p[1] = -w_m[0];
                if (dim == 2) {
                    // Copy out axial velocity
                    w_p[2] = w_m[2];
                }
                // Copy out energy
                w_p[dim + 1] = w_m[dim + 1];
            } else if (_bc_map.is_wall(boundary_id)) {
                w_p[0] = w_m[0];
                for (unsigned int d = 0; d < dim; d++) {
                    w_p[d + 1] = w_m[d + 1] - 2.0 * r_rho_u_dot_n * normal[d];
                }
                w_p[dim + 1] = w_m[dim + 1];
            } else
                AssertThrow(false,
                            ExcMessage("Unknown boundary id, did "
                                       "you set a boundary condition for "
                                       "this part of the domain boundary?"));

            const auto r = phi.quadrature_point(q);
            auto flux = radial_euler_numerical_flux<dim>(w_m, w_p, normal, r,
                                                         gamma, false);

            if (_bc_map.is_axial(boundary_id)) {
                for (unsigned int c = 0; c < dim + 2; c++) {
                    flux[c] = 0.0;
                }
            }

            if (at_outflow) {
                for (unsigned int v = 0; v < VectorizedArray<Number>::size();
                     ++v) {
                    if (r_rho_u_dot_n[v] < -1e-12) {
                        for (unsigned int d = 0; d < 1; ++d) {
                            flux[d + 1][v] = 0.;
                        }
                    }
                }
            } else {
                // std::cout << "inflow flux: " << flux << std::endl;
            }

            phi.submit_value(-flux, q);
        }

        phi.integrate_scatter(EvaluationFlags::values, dst);
    }
}

template <int dim, int degree, int n_points_1d>
void RadialEulerOperator<dim, degree, n_points_1d>::
    local_apply_inverse_mass_matrix(
        const MatrixFree<dim, Number> &,
        LinearAlgebra::distributed::Vector<Number> &dst,
        const LinearAlgebra::distributed::Vector<Number> &src,
        const std::pair<unsigned int, unsigned int> &cell_range) const {
    FEEvaluation<dim, degree, degree + 1, dim + 2, Number> phi(data, 0, 1);
    MatrixFreeOperators::CellwiseInverseMassMatrix<dim, degree, dim + 2, Number>
        inverse(phi);

    for (unsigned int cell = cell_range.first; cell < cell_range.second;
         ++cell) {
        phi.reinit(cell);
        phi.read_dof_values(src);

        inverse.apply(phi.begin_dof_values(), phi.begin_dof_values());

        phi.set_dof_values(dst);
    }
}

template <int dim, int degree, int n_points_1d>
void RadialEulerOperator<dim, degree, n_points_1d>::perform_stage(
    const Number current_time, const Number factor_solution,
    const Number factor_ai,
    const LinearAlgebra::distributed::Vector<Number> &current_ri,
    LinearAlgebra::distributed::Vector<Number> &vec_ki,
    LinearAlgebra::distributed::Vector<Number> &solution,
    LinearAlgebra::distributed::Vector<Number> &next_ri) const {
    {
        TimerOutput::Scope t(timer, "rk_stage - integrals L_h");

        for (auto &i : bc_map().inflow_boundaries()) i.second->set_time(current_time);
        for (auto &i : bc_map().subsonic_outflow_boundaries())
            i.second->set_time(current_time);

        data.loop(&RadialEulerOperator::local_apply_cell,
                  &RadialEulerOperator::local_apply_face,
                  &RadialEulerOperator::local_apply_boundary_face, this, vec_ki,
                  current_ri, true,
                  MatrixFree<dim, Number>::DataAccessOnFaces::values,
                  MatrixFree<dim, Number>::DataAccessOnFaces::values);
    }

    {
        TimerOutput::Scope t(timer, "rk_stage - inv mass + vec upd");
        data.cell_loop(
            &RadialEulerOperator::local_apply_inverse_mass_matrix, this,
            next_ri, vec_ki,
            std::function<void(const unsigned int, const unsigned int)>(),
            [&](const unsigned int start_range, const unsigned int end_range) {
                const Number ai = factor_ai;
                const Number bi = factor_solution;
                if (ai == Number()) {
                    /* DEAL_II_OPENMP_SIMD_PRAGMA */
                    for (unsigned int i = start_range; i < end_range; ++i) {
                        const Number k_i = next_ri.local_element(i);
                        const Number sol_i = solution.local_element(i);
                        solution.local_element(i) = sol_i + bi * k_i;
                    }
                } else {
                    /* DEAL_II_OPENMP_SIMD_PRAGMA */
                    for (unsigned int i = start_range; i < end_range; ++i) {
                        const Number k_i = next_ri.local_element(i);
                        const Number sol_i = solution.local_element(i);
                        solution.local_element(i) = sol_i + bi * k_i;
                        next_ri.local_element(i) = sol_i + ai * k_i;
                    }
                }
            });
    }
}

template <int dim, int degree, int n_points_1d>
void RadialEulerOperator<dim, degree, n_points_1d>::project(
    const Function<dim> &function,
    LinearAlgebra::distributed::Vector<Number> &solution) const {
    FEEvaluation<dim, degree, degree + 1, dim + 2, Number> phi(data, 0, 1);
    MatrixFreeOperators::CellwiseInverseMassMatrix<dim, degree, dim + 2, Number>
        inverse(phi);
    solution.zero_out_ghost_values();
    for (unsigned int cell = 0; cell < data.n_cell_batches(); ++cell) {
        phi.reinit(cell);
        for (const unsigned int q : phi.quadrature_point_indices())
            phi.submit_dof_value(
                evaluate_function<dim, Number, VectorizedArray<Number>>(function, phi.quadrature_point(q)), q);
        inverse.transform_from_q_points_to_basis(
            dim + 2, phi.begin_dof_values(), phi.begin_dof_values());
        phi.set_dof_values(solution);
    }
}

template <int dim, int degree, int n_points_1d>
double
RadialEulerOperator<dim, degree, n_points_1d>::compute_cell_transport_speed(
    const LinearAlgebra::distributed::Vector<Number> &solution) const {
    TimerOutput::Scope t(timer, "compute transport speed");
    Number max_transport = 0;
    FEEvaluation<dim, degree, degree + 1, dim + 2, Number> phi(data, 0, 1);

    for (unsigned int cell = 0; cell < data.n_cell_batches(); ++cell) {
        phi.reinit(cell);
        phi.gather_evaluate(solution, EvaluationFlags::values);
        VectorizedArray<Number> local_max = 0.;
        for (const unsigned int q : phi.quadrature_point_indices()) {
            const auto solution = phi.get_value(q);
            const auto velocity = radial_euler_velocity<dim>(solution);
            const auto pressure = radial_euler_pressure<dim>(solution, gamma);

            const auto inverse_jacobian = phi.inverse_jacobian(q);
            const auto convective_speed = inverse_jacobian * velocity;
            VectorizedArray<Number> convective_limit = 0.;
            for (unsigned int d = 0; d < dim; ++d)
                convective_limit =
                    std::max(convective_limit, std::abs(convective_speed[d]));

            const auto speed_of_sound =
                std::sqrt(gamma * pressure * (1. / solution[0]));

            Tensor<1, dim, VectorizedArray<Number>> eigenvector;
            for (unsigned int d = 0; d < dim; ++d) eigenvector[d] = 1.;
            for (unsigned int i = 0; i < 5; ++i) {
                eigenvector = transpose(inverse_jacobian) *
                              (inverse_jacobian * eigenvector);
                VectorizedArray<Number> eigenvector_norm = 0.;
                for (unsigned int d = 0; d < dim; ++d)
                    eigenvector_norm =
                        std::max(eigenvector_norm, std::abs(eigenvector[d]));
                eigenvector /= eigenvector_norm;
            }
            const auto jac_times_ev = inverse_jacobian * eigenvector;
            const auto max_eigenvalue = std::sqrt(
                (jac_times_ev * jac_times_ev) / (eigenvector * eigenvector));
            local_max = std::max(
                local_max, max_eigenvalue * speed_of_sound + convective_limit);
        }

        for (unsigned int v = 0; v < data.n_active_entries_per_cell_batch(cell);
             ++v)
            for (unsigned int d = 0; d < 3; ++d)
                max_transport = std::max(max_transport, local_max[v]);
    }

    max_transport = Utilities::MPI::max(max_transport, MPI_COMM_WORLD);

    return max_transport;
}
}  // namespace CylindricalEuler
