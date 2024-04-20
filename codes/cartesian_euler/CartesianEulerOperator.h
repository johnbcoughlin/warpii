#pragma once

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/time_stepping.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/types.h>
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

#include "cartesian_euler.h"
#include "euler/bc_helper.h"
#include "function_eval.h"

namespace CartesianEuler {
using namespace dealii;

using Number = double;
using VA = VectorizedArray<Number, 1>;

template <int dim, int degree, int n_points_1d> class CartesianEulerOperator {

public:
  CartesianEulerOperator(TimerOutput &timer, double gamma)
      : timer(timer), gamma(gamma) {}

  void reinit(const Mapping<dim> &mapping, const DoFHandler<dim> &dof_handler);

  void set_inflow_boundary(const types::boundary_id boundary_id,
                           std::unique_ptr<Function<dim>> inflow_function);
  void
  set_subsonic_outflow_boundary(const types::boundary_id boundary_id,
                                std::unique_ptr<Function<dim>> outflow_energy);
  void set_supersonic_outflow_boundary(const types::boundary_id boundary_id);
  void set_axial_boundary(const types::boundary_id boundary_id);
  void set_wall_boundary(const types::boundary_id boundary_id);

  void
  perform_stage(const Number cur_time, const Number factor_solution,
                const Number factor_ai,
                const LinearAlgebra::distributed::Vector<Number> &current_ri,
                LinearAlgebra::distributed::Vector<Number> &vec_ki,
                LinearAlgebra::distributed::Vector<Number> &solution,
                LinearAlgebra::distributed::Vector<Number> &next_ri) const;

  void perform_forward_euler_step(
      LinearAlgebra::distributed::Vector<Number> &dst,
      const LinearAlgebra::distributed::Vector<Number> &src,
      std::vector<LinearAlgebra::distributed::Vector<Number>> &sol_registers,
      const double dt, const double t, double alpha = 1.0,
      double beta = 0.0) const;

  void
  initialize_vector(LinearAlgebra::distributed::Vector<Number> &vector) const;

  void project(const Function<dim> &function,
               LinearAlgebra::distributed::Vector<Number> &solution) const;

  double compute_cell_transport_speed(
      const LinearAlgebra::distributed::Vector<Number> &solution) const;

  const EulerBCMap<dim> bc_map() const { return _bc_map; }

  EulerBCMap<dim> &bc_map() { return _bc_map; }

private:
  MatrixFree<dim, Number, VA> data;
  TimerOutput &timer;
  double gamma;
  EulerBCMap<dim> _bc_map;

  void local_apply_inverse_mass_matrix(
      const MatrixFree<dim, Number, VA> &data,
      LinearAlgebra::distributed::Vector<Number> &dst,
      const LinearAlgebra::distributed::Vector<Number> &src,
      const std::pair<unsigned int, unsigned int> &cell_range) const;

  void local_apply_cell(
      const MatrixFree<dim, Number, VA> &data,
      LinearAlgebra::distributed::Vector<Number> &dst,
      const LinearAlgebra::distributed::Vector<Number> &src,
      const std::pair<unsigned int, unsigned int> &cell_range) const;

  void local_apply_face(
      const MatrixFree<dim, Number, VA> &data,
      LinearAlgebra::distributed::Vector<Number> &dst,
      const LinearAlgebra::distributed::Vector<Number> &src,
      const std::pair<unsigned int, unsigned int> &face_range) const;

  void local_apply_boundary_face(
      const MatrixFree<dim, Number, VA> &data,
      LinearAlgebra::distributed::Vector<Number> &dst,
      const LinearAlgebra::distributed::Vector<Number> &src,
      const std::pair<unsigned int, unsigned int> &face_range) const;

  void local_apply_positivity_limiter(
      const MatrixFree<dim, Number, VA> &,
      LinearAlgebra::distributed::Vector<Number> &dst,
      const LinearAlgebra::distributed::Vector<Number> &src,
      const std::pair<unsigned int, unsigned int> &cell_range) const;
};

template <int dim, int degree, int n_points_1d>
void CartesianEulerOperator<dim, degree, n_points_1d>::reinit(
    const Mapping<dim> &mapping, const DoFHandler<dim> &dof_handler) {
  const std::vector<const DoFHandler<dim> *> dof_handlers = {&dof_handler};
  const AffineConstraints<double> dummy;
  const std::vector<const AffineConstraints<double> *> constraints = {&dummy};

  const std::vector<Quadrature<1>> quadratures = {
      QGauss<1>(n_points_1d),
      QGauss<1>(degree + 1),
      QGaussLobatto<1>(degree + 1),
  };

  typename MatrixFree<dim, Number, VA>::AdditionalData additional_data;
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
      MatrixFree<dim, Number, VA>::AdditionalData::none;

  data.reinit(mapping, dof_handlers, constraints, quadratures, additional_data);
}

template <int dim, int degree, int n_points_1d>
void CartesianEulerOperator<dim, degree, n_points_1d>::initialize_vector(
    LinearAlgebra::distributed::Vector<Number> &vector) const {
  data.initialize_dof_vector(vector);
}

template <int dim, int degree, int n_points_1d>
void CartesianEulerOperator<dim, degree, n_points_1d>::local_apply_cell(
    const MatrixFree<dim, Number, VA> &,
    LinearAlgebra::distributed::Vector<Number> &dst,
    const LinearAlgebra::distributed::Vector<Number> &src,
    const std::pair<unsigned int, unsigned int> &cell_range) const {
  FEEvaluation<dim, degree, n_points_1d, dim + 2, Number, VA> phi(data);

  for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
    phi.reinit(cell);
    phi.gather_evaluate(src, EvaluationFlags::values);

    for (const unsigned int q : phi.quadrature_point_indices()) {
      const auto w_q = phi.get_value(q);
      phi.submit_gradient(euler_flux<dim, VA>(w_q, gamma), q);
    }

    phi.integrate_scatter(EvaluationFlags::gradients, dst);
  }
}

template <int dim, int degree, int n_points_1d>
void CartesianEulerOperator<dim, degree, n_points_1d>::local_apply_face(
    const MatrixFree<dim, Number, VA> &,
    LinearAlgebra::distributed::Vector<Number> &dst,
    const LinearAlgebra::distributed::Vector<Number> &src,
    const std::pair<unsigned int, unsigned int> &face_range) const {
  FEFaceEvaluation<dim, degree, n_points_1d, dim + 2, Number, VA> phi_m(data,
                                                                        true);
  FEFaceEvaluation<dim, degree, n_points_1d, dim + 2, Number, VA> phi_p(data,
                                                                        false);

  for (unsigned int face = face_range.first; face < face_range.second; ++face) {
    phi_p.reinit(face);
    phi_p.gather_evaluate(src, EvaluationFlags::values);

    phi_m.reinit(face);
    phi_m.gather_evaluate(src, EvaluationFlags::values);

    for (const unsigned int q : phi_m.quadrature_point_indices()) {
      const auto numerical_flux =
          euler_numerical_flux<dim>(phi_m.get_value(q), phi_p.get_value(q),
                                    phi_m.normal_vector(q), gamma);
      phi_m.submit_value(-numerical_flux, q);
      phi_p.submit_value(numerical_flux, q);
    }

    phi_p.integrate_scatter(EvaluationFlags::values, dst);
    phi_m.integrate_scatter(EvaluationFlags::values, dst);
  }
}

template <int dim, int degree, int n_points_1d>
void CartesianEulerOperator<dim, degree, n_points_1d>::
    local_apply_boundary_face(
        const MatrixFree<dim, Number, VA> &,
        LinearAlgebra::distributed::Vector<Number> &dst,
        const LinearAlgebra::distributed::Vector<Number> &src,
        const std::pair<unsigned int, unsigned int> &face_range) const {
  FEFaceEvaluation<dim, degree, n_points_1d, dim + 2, Number, VA> phi(data,
                                                                      true);

  for (unsigned int face = face_range.first; face < face_range.second; ++face) {
    phi.reinit(face);
    phi.gather_evaluate(src, EvaluationFlags::values);

    const auto boundary_id = data.get_boundary_id(face);

    for (const unsigned int q : phi.quadrature_point_indices()) {
      const auto w_m = phi.get_value(q);
      const auto normal = phi.normal_vector(q);

      auto rho_u_dot_n = w_m[1] * normal[0];
      for (unsigned int d = 1; d < dim; d++) {
        rho_u_dot_n += w_m[1 + d] * normal[d];
      }

      bool at_outflow = false;
      Tensor<1, dim + 2, VA> w_p;
      if (_bc_map.is_inflow(boundary_id)) {
        w_p = evaluate_function<dim, Number, VA>(
            *_bc_map.get_inflow(boundary_id), phi.quadrature_point(q));
      } else if (_bc_map.is_subsonic_outflow(boundary_id)) {
        w_p = w_m;
        w_p[dim + 1] = evaluate_function<dim, Number, VA>(
            *_bc_map.get_subsonic_outflow_energy(boundary_id),
            phi.quadrature_point(q), dim + 1);
        at_outflow = true;
      } else if (_bc_map.is_supersonic_outflow(boundary_id)) {
        w_p = w_m;
        at_outflow = true;
      } else if (_bc_map.is_wall(boundary_id)) {
        // Copy out density
        w_p[0] = w_m[0];
        for (unsigned int d = 0; d < dim; d++) {
          w_p[d + 1] = w_m[d + 1] - 2.0 * rho_u_dot_n * normal[d];
        }
        w_p[dim + 1] = w_m[dim + 1];
      } else {
        AssertThrow(
            false,
            ExcMessage("Unknown boundary id, did you set a boundary condition "
                       "for this part of the domain boundary?"));
      }

      auto flux = euler_numerical_flux<dim, VA>(w_m, w_p, normal, gamma);

      if (at_outflow) {
        for (unsigned int v = 0; v < VA::size(); ++v) {
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

template <int dim, int degree, int n_points_1d>
void CartesianEulerOperator<dim, degree, n_points_1d>::
    local_apply_inverse_mass_matrix(
        const MatrixFree<dim, Number, VA> &,
        LinearAlgebra::distributed::Vector<Number> &dst,
        const LinearAlgebra::distributed::Vector<Number> &src,
        const std::pair<unsigned int, unsigned int> &cell_range) const {
  FEEvaluation<dim, degree, degree + 1, dim + 2, Number, VA> phi(data, 0, 1);
  MatrixFreeOperators::CellwiseInverseMassMatrix<dim, degree, dim + 2, Number,
                                                 VA>
      inverse(phi);

  for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
    phi.reinit(cell);
    phi.read_dof_values(src);

    inverse.apply(phi.begin_dof_values(), phi.begin_dof_values());

    phi.set_dof_values(dst);
  }
}

template <int dim, int degree, int n_points_1d>
void CartesianEulerOperator<dim, degree, n_points_1d>::
    local_apply_positivity_limiter(
        const MatrixFree<dim, Number, VA> &,
        LinearAlgebra::distributed::Vector<Number> &dst,
        const LinearAlgebra::distributed::Vector<Number> &src,
        const std::pair<unsigned int, unsigned int> &cell_range) const {
  // Used only for the area calculation
  FEEvaluation<dim, degree, degree + 1, 1, Number, VA> phi_scalar(data, 0, 1);
  // This should be constructed from the quadrature rule used for the
  // positivity-preserving step.
  // TODO: programmatically determine the number of quadrature points
  FEEvaluation<dim, degree, degree + 1, dim + 2, Number, VA> phi(data, 0, 2);
  MatrixFreeOperators::CellwiseInverseMassMatrix<dim, degree, dim + 2, Number,
                                                 VA>
      inverse(phi);

  for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
    phi_scalar.reinit(cell);
    phi.reinit(cell);

    phi.gather_evaluate(src, EvaluationFlags::values);
    VA rho_min = VA(std::numeric_limits<Number>::infinity());

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
    auto p_bar = euler_pressure<dim>(cell_avg, gamma);

    for (unsigned int v = 0; v < VA::size(); ++v) {
      if (rho_bar[v] <= 0.0) {
        AssertThrow(false, ExcMessage("Cell average density was negative"));
      }
      if (p_bar[v] <= 0.0) {
        AssertThrow(false, ExcMessage("Cell average pressure was negative"));
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
    VA p_min = VA(std::numeric_limits<Number>::infinity());
    for (const unsigned int q : phi.quadrature_point_indices()) {
      auto v = phi.get_value(q);
      v[0] = theta_rho * (v[0] - rho_bar) + rho_bar;
      p_min = std::min(euler_pressure<dim>(v, gamma), p_min);
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
        // theta_E_v << std::endl; std::cout << "theta_rho: " << theta_rho_v <<
        // "; denom_rho: " << denom_rho[v] << std::endl; std::cout << "min_rho:
        // " << rho_min[v] << "; rho_bar: " << cell_avg[0][v] << std::endl;
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
      auto pressure = euler_pressure<dim>(v, gamma);
      for (unsigned int vec_i = 0; vec_i < VA::size(); ++vec_i) {
        // AssertThrow(v[dim+2][vec_i] > 1e-12, ExcMessage("Submitting negative
        // density to quad point"));
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
        AssertThrow(rho[vec_i] > 1e-12,
                    ExcMessage("Submitting negative density to quad point"));
        AssertThrow(pressure[vec_i] > 1e-12,
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

template <int dim, int degree, int n_points_1d>
void CartesianEulerOperator<dim, degree, n_points_1d>::perform_stage(
    const Number current_time, const Number factor_solution,
    const Number factor_ai,
    const LinearAlgebra::distributed::Vector<Number> &current_ri,
    LinearAlgebra::distributed::Vector<Number> &vec_ki,
    LinearAlgebra::distributed::Vector<Number> &solution,
    LinearAlgebra::distributed::Vector<Number> &next_ri) const {
  {
    TimerOutput::Scope t(timer, "rk_stage - integrals L_h");

    for (auto &i : _bc_map.inflow_boundaries())
      i.second->set_time(current_time);
    for (auto &i : _bc_map.subsonic_outflow_boundaries())
      i.second->set_time(current_time);

    data.loop(&CartesianEulerOperator::local_apply_cell,
              &CartesianEulerOperator::local_apply_face,
              &CartesianEulerOperator::local_apply_boundary_face, this, vec_ki,
              current_ri, true,
              MatrixFree<dim, Number, VA>::DataAccessOnFaces::values,
              MatrixFree<dim, Number, VA>::DataAccessOnFaces::values);
  }

  {
    TimerOutput::Scope t(timer, "rk_stage - inv mass + vec upd");
    data.cell_loop(
        &CartesianEulerOperator::local_apply_inverse_mass_matrix, this, next_ri,
        vec_ki, std::function<void(const unsigned int, const unsigned int)>(),
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
void CartesianEulerOperator<dim, degree, n_points_1d>::
    perform_forward_euler_step(
        LinearAlgebra::distributed::Vector<Number> &dst,
        const LinearAlgebra::distributed::Vector<Number> &u,
        std::vector<LinearAlgebra::distributed::Vector<Number>> &sol_registers,
        const double dt, const double t, double alpha, double beta) const {

  auto Mdudt_register = sol_registers.at(0);
  auto dudt_register = sol_registers.at(1);
  auto sol_before_limiting = sol_registers.at(2);

  {
    TimerOutput::Scope timer_scope(timer, "rk_stage - integrals L_h");

    for (auto &i : _bc_map.inflow_boundaries())
      i.second->set_time(t);
    for (auto &i : _bc_map.subsonic_outflow_boundaries())
      i.second->set_time(t);

    data.loop(&CartesianEulerOperator::local_apply_cell,
              &CartesianEulerOperator::local_apply_face,
              &CartesianEulerOperator::local_apply_boundary_face, this,
              Mdudt_register, u, true,
              MatrixFree<dim, Number, VA>::DataAccessOnFaces::values,
              MatrixFree<dim, Number, VA>::DataAccessOnFaces::values);
  }

  {
    TimerOutput::Scope t(timer, "rk_stage - inv mass + vec upd");
    data.cell_loop(
        &CartesianEulerOperator::local_apply_inverse_mass_matrix, this,
        dudt_register, Mdudt_register,
        std::function<void(const unsigned int, const unsigned int)>(),
        [&](const unsigned int start_range, const unsigned int end_range) {
          /* DEAL_II_OPENMP_SIMD_PRAGMA */
          for (unsigned int i = start_range; i < end_range; ++i) {
            const Number dudt_i = dudt_register.local_element(i);
            const Number dst_i = dst.local_element(i);
            const Number u_i = u.local_element(i);
            sol_before_limiting.local_element(i) =
                beta * dst_i + alpha * (u_i + dt * dudt_i);
          }
        });
  }

  {
    TimerOutput::Scope t(timer, "rk stage - positivity limiter");
    data.cell_loop(&CartesianEulerOperator::local_apply_positivity_limiter,
                   this, dst, sol_before_limiting);
  }

  // dst.sadd(0.0, sol_before_limiting);
}

template <int dim, int degree, int n_points_1d>
void CartesianEulerOperator<dim, degree, n_points_1d>::project(
    const Function<dim> &function,
    LinearAlgebra::distributed::Vector<Number> &solution) const {
  FEEvaluation<dim, degree, degree + 1, dim + 2, Number, VA> phi(data, 0, 1);
  MatrixFreeOperators::CellwiseInverseMassMatrix<dim, degree, dim + 2, Number,
                                                 VA>
      inverse(phi);
  solution.zero_out_ghost_values();
  for (unsigned int cell = 0; cell < data.n_cell_batches(); ++cell) {
    phi.reinit(cell);
    for (const unsigned int q : phi.quadrature_point_indices())
      phi.submit_dof_value(
          evaluate_function<dim, Number, VA>(function, phi.quadrature_point(q)),
          q);
    inverse.transform_from_q_points_to_basis(dim + 2, phi.begin_dof_values(),
                                             phi.begin_dof_values());
    phi.set_dof_values(solution);
  }
}

template <int dim, int degree, int n_points_1d>
double
CartesianEulerOperator<dim, degree, n_points_1d>::compute_cell_transport_speed(
    const LinearAlgebra::distributed::Vector<Number> &solution) const {
  TimerOutput::Scope t(timer, "compute transport speed");
  Number max_transport = 0;
  FEEvaluation<dim, degree, degree + 1, dim + 2, Number, VA> phi(data, 0, 1);

  for (unsigned int cell = 0; cell < data.n_cell_batches(); ++cell) {
    phi.reinit(cell);
    phi.gather_evaluate(solution, EvaluationFlags::values);
    VA local_max = 0.;
    for (const unsigned int q : phi.quadrature_point_indices()) {
      const auto solution = phi.get_value(q);
      const auto velocity = euler_velocity<dim>(solution);
      const auto pressure = euler_pressure<dim>(solution, gamma);

      const auto inverse_jacobian = phi.inverse_jacobian(q);
      const auto convective_speed = inverse_jacobian * velocity;
      VA convective_limit = 0.;
      for (unsigned int d = 0; d < dim; ++d)
        convective_limit =
            std::max(convective_limit, std::abs(convective_speed[d]));

      const auto speed_of_sound =
          std::sqrt(gamma * pressure * (1. / solution[0]));

      Tensor<1, dim, VA> eigenvector;
      for (unsigned int d = 0; d < dim; ++d)
        eigenvector[d] = 1.;
      for (unsigned int i = 0; i < 5; ++i) {
        eigenvector =
            transpose(inverse_jacobian) * (inverse_jacobian * eigenvector);
        VA eigenvector_norm = 0.;
        for (unsigned int d = 0; d < dim; ++d)
          eigenvector_norm =
              std::max(eigenvector_norm, std::abs(eigenvector[d]));
        eigenvector /= eigenvector_norm;
      }
      const auto jac_times_ev = inverse_jacobian * eigenvector;
      const auto max_eigenvalue = std::sqrt((jac_times_ev * jac_times_ev) /
                                            (eigenvector * eigenvector));
      local_max = std::max(local_max,
                           max_eigenvalue * speed_of_sound + convective_limit);
    }

    for (unsigned int v = 0; v < data.n_active_entries_per_cell_batch(cell);
         ++v) {
      for (unsigned int d = 0; d < 3; ++d)
        max_transport = std::max(max_transport, local_max[v]);
    }
  }

  max_transport = Utilities::MPI::max(max_transport, MPI_COMM_WORLD);

  return max_transport;
}
} // namespace CartesianEuler
