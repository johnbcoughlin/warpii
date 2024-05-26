#pragma once

#include <deal.II/base/array_view.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/utilities.h>
#include <deal.II/fe/fe_series.h>
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/fe/fe_values_extractors.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/vector.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <algorithm>

#include "../dof_utils.h"
#include "../utilities.h"
#include "dg_discretization.h"
#include "euler.h"
#include "solution_vec.h"
#include "species.h"

namespace warpii {
namespace five_moment {

using namespace dealii;

template <int dim>
struct ScratchData {
    ScratchData(
        const std::shared_ptr<FiveMomentDGDiscretization<dim>> discretization)
        : fe_values(discretization->get_mapping(), discretization->get_fe(),
                    QGaussLobatto<dim>(), UpdateFlags::update_values) {}

    FEValues<dim> fe_values;
    std::vector<std::vector<double>> u_values;
};

struct CopyData {};

template <int dim>
class FluidFluxESDGSEMOperator {
   public:
    FluidFluxESDGSEMOperator(
        std::shared_ptr<FiveMomentDGDiscretization<dim>> discretization,
        double gas_gamma, std::vector<std::shared_ptr<Species<dim>>> species)
        : discretization(discretization),
          gas_gamma(gas_gamma),
          n_species(species.size()),
          species(species) {}

    void perform_forward_euler_step(
        FiveMSolutionVec &dst, const FiveMSolutionVec &u,
        std::vector<FiveMSolutionVec> &sol_registers, const double dt,
        const double t, const double alpha = 1.0,
        const double beta = 0.0) const;

    double recommend_dt(const MatrixFree<dim, double> &mf,
                        const FiveMSolutionVec &sol);

   private:
    void local_apply_inverse_mass_matrix(
        const MatrixFree<dim, double> &mf,
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
        const MatrixFree<dim, double> &mf,
        LinearAlgebra::distributed::Vector<double> &dst,
        const LinearAlgebra::distributed::Vector<double> &src,
        const std::pair<unsigned int, unsigned int> &face_range,
        FiveMBoundaryIntegratedFluxesVector &boundary_integrated_fluxes) const;

    double compute_cell_transport_speed(
        const MatrixFree<dim, double> &mf,
        const LinearAlgebra::distributed::Vector<double> &sol) const;
    /**
     * Compute the troubled cell indicator of Persson and Peraire,
     * "Sub-Cell Shock Capturing for Discontinuous Galerkin Methods"
     *
     * @param phi: Should have already been reinited for the current cell
     */
    VectorizedArray<double> shock_indicators(
        const FEEvaluation<dim, -1, 0, dim + 2, double> &phi,
        FESeries::Legendre<dim> &legendre) const;

    void calculate_high_order_EC_flux(
        LinearAlgebra::distributed::Vector<double> &dst,
        FEEvaluation<dim, -1, 0, dim + 2, double> &phi,
        const FEEvaluation<dim, -1, 0, dim + 2, double> &phi_reader,
        const FullMatrix<double> &D, unsigned int d,
        VectorizedArray<double> alpha, bool log = false) const;

    void calculate_first_order_ES_flux(
        LinearAlgebra::distributed::Vector<double> &dst,
        FEEvaluation<dim, -1, 0, dim + 2, double> &phi,
        const FEEvaluation<dim, -1, 0, dim + 2, double> &phi_reader,
        const std::vector<double> &quadrature_weights,
        const FullMatrix<double> &Q, unsigned int d,
        VectorizedArray<double> alpha, bool log = false) const;

    std::shared_ptr<FiveMomentDGDiscretization<dim>> discretization;
    double gas_gamma;
    unsigned int n_species;
    std::vector<std::shared_ptr<Species<dim>>> species;
};

template <int dim>
void FluidFluxESDGSEMOperator<dim>::perform_forward_euler_step(
    FiveMSolutionVec &dst, const FiveMSolutionVec &u,
    std::vector<FiveMSolutionVec> &sol_registers, const double dt,
    const double t, const double alpha, const double beta) const {
    using Iterator = typename DoFHandler<1>::active_cell_iterator;

    auto Mdudt_register = sol_registers.at(0);
    auto dudt_register = sol_registers.at(1);
    auto sol_before_limiting = sol_registers.at(2);

    {
        for (auto sp : species) {
            for (auto &i : sp->bc_map.inflow_boundaries()) {
                i.second->set_time(t);
            }
        }

        std::function<void(const MatrixFree<dim, Number> &,
                           LinearAlgebra::distributed::Vector<double> &,
                           const LinearAlgebra::distributed::Vector<double> &,
                           const std::pair<unsigned int, unsigned int> &)>
            cell_operation =
                [&](const MatrixFree<dim, Number> &mf,
                    LinearAlgebra::distributed::Vector<double> &dst,
                    const LinearAlgebra::distributed::Vector<double> &src,
                    const std::pair<unsigned int, unsigned int> &cell_range)
            -> void { this->local_apply_cell(mf, dst, src, cell_range); };
        std::function<void(const MatrixFree<dim, Number> &,
                           LinearAlgebra::distributed::Vector<double> &,
                           const LinearAlgebra::distributed::Vector<double> &,
                           const std::pair<unsigned int, unsigned int> &)>
            face_operation =
                [&](const MatrixFree<dim, Number> &mf,
                    LinearAlgebra::distributed::Vector<double> &dst,
                    const LinearAlgebra::distributed::Vector<double> &src,
                    const std::pair<unsigned int, unsigned int> &cell_range)
            -> void { this->local_apply_face(mf, dst, src, cell_range); };
        FiveMBoundaryIntegratedFluxesVector &d_dt_boundary_integrated_fluxes =
            dudt_register.boundary_integrated_fluxes;
        std::function<void(const MatrixFree<dim, Number> &,
                           LinearAlgebra::distributed::Vector<double> &,
                           const LinearAlgebra::distributed::Vector<double> &,
                           const std::pair<unsigned int, unsigned int> &)>
            boundary_operation =
                [this, &d_dt_boundary_integrated_fluxes](
                    const MatrixFree<dim, Number> &mf,
                    LinearAlgebra::distributed::Vector<double> &dst,
                    const LinearAlgebra::distributed::Vector<double> &src,
                    const std::pair<unsigned int, unsigned int> &cell_range)
            -> void {
            this->local_apply_boundary_face(mf, dst, src, cell_range,
                                            d_dt_boundary_integrated_fluxes);
        };

        discretization->mf.loop(
            cell_operation, face_operation, boundary_operation,
            Mdudt_register.mesh_sol, u.mesh_sol, true,
            MatrixFree<dim, double>::DataAccessOnFaces::values,
            MatrixFree<dim, double>::DataAccessOnFaces::values);
    }

    {
        discretization->mf.cell_loop(
            &FluidFluxESDGSEMOperator<dim>::local_apply_inverse_mass_matrix,
            this, dudt_register.mesh_sol, Mdudt_register.mesh_sol,
            std::function<void(const unsigned int, const unsigned int)>(),
            [&](const unsigned int start_range, const unsigned int end_range) {
                /* DEAL_II_OPENMP_SIMD_PRAGMA */
                for (unsigned int i = start_range; i < end_range; ++i) {
                    const double dudt_i =
                        dudt_register.mesh_sol.local_element(i);
                    const double dst_i = dst.mesh_sol.local_element(i);
                    const double u_i = u.mesh_sol.local_element(i);
                    dst.mesh_sol.local_element(i) =
                        beta * dst_i + alpha * (u_i + dt * dudt_i);
                }
            });
        // dst = beta * dest + alpha * (u + dt * dudt)
        if (!dst.boundary_integrated_fluxes.is_empty()) {
            dst.boundary_integrated_fluxes.sadd(
                beta, alpha * dt, dudt_register.boundary_integrated_fluxes);
            dst.boundary_integrated_fluxes.sadd(1.0, alpha,
                                                u.boundary_integrated_fluxes);
        }
    }
}

template <int dim>
void FluidFluxESDGSEMOperator<dim>::local_apply_inverse_mass_matrix(
    const MatrixFree<dim, double> &mf,
    LinearAlgebra::distributed::Vector<double> &dst,
    const LinearAlgebra::distributed::Vector<double> &src,
    const std::pair<unsigned int, unsigned int> &cell_range) const {
    for (unsigned int species_index = 0; species_index < n_species;
         species_index++) {
        unsigned int first_component = species_index * (dim + 2);
        FEEvaluation<dim, -1, 0, dim + 2, double> phi(mf, 0, 1,
                                                      first_component);
        MatrixFreeOperators::CellwiseInverseMassMatrix<dim, -1, dim + 2, double>
            inverse(phi);

        for (unsigned int cell = cell_range.first; cell < cell_range.second;
             ++cell) {
            phi.reinit(cell);
            phi.read_dof_values(src);

            inverse.apply(phi.begin_dof_values(), phi.begin_dof_values());

            phi.set_dof_values(dst);
        }
    }
}

template <int dim>
VectorizedArray<double> jacobian_determinant(
    const FEEvaluation<dim, -1, 0, dim + 2, double> &phi, unsigned int q) {
    Tensor<2, dim, VectorizedArray<double>> Jinv_j = phi.inverse_jacobian(q);
    VectorizedArray<double> Jdet_j = 1.0 / warpii::determinant(Jinv_j);
    return Jdet_j;
}

/**
 * Computes the contravariant basis vector Ja^d at the quadrature point q,
 * where d is the dimension index.
 *
 * For details on what we're doing here, see Winters et al. (2020),
 * "Construction of Modern Robust Nodal Discontinuous Galerkin Spectral Element
 * Methods for the Compressible Navier-Stokes Equations", Section 4.3
 *
 * This is equation (172), taking advantage of the fact that
 * `FEEvaluation::inverse_jacobian` returns precisely the matrix whose columns
 * are \vec{a}^i. We then multiply by the Jacobian determinant J.
 */
template <int dim>
Tensor<1, dim, VectorizedArray<double>> scaled_contravariant_basis_vector(
    const FEEvaluation<dim, -1, 0, dim + 2, double> &phi, unsigned int q,
    unsigned int d) {
    Tensor<2, dim, VectorizedArray<double>> Jinv_j = phi.inverse_jacobian(q);
    VectorizedArray<double> Jdet_j = jacobian_determinant(phi, q);

    return Jdet_j * tensor_column(Jinv_j, d);
}

template <int dim>
void FluidFluxESDGSEMOperator<dim>::calculate_high_order_EC_flux(
    LinearAlgebra::distributed::Vector<double> &dst,
    FEEvaluation<dim, -1, 0, dim + 2, double> &phi,
    const FEEvaluation<dim, -1, 0, dim + 2, double> &phi_reader,
    const FullMatrix<double> &D, unsigned int d, VectorizedArray<double> alpha,
    bool /* log */) const {
    unsigned int fe_degree = discretization->get_fe_degree();
    unsigned int Np = fe_degree + 1;

    for (const unsigned int qj : phi.quadrature_point_indices()) {
        auto Jdet_j = jacobian_determinant(phi, qj);
        auto Jai_j = scaled_contravariant_basis_vector(phi, qj, d);

        unsigned int j = quad_point_1d_index<dim>(qj, Np, d);
        auto uj = phi_reader.get_value(qj);
        Tensor<1, dim + 2, VectorizedArray<double>> flux_j;
        for (unsigned int di = 0; di < dim; di++) {
            flux_j[di] = 0.0;
        }

        for (unsigned int l = 0; l < Np; l++) {
            unsigned int ql = quadrature_point_neighbor<dim>(qj, l, Np, d);
            auto Jai_l = scaled_contravariant_basis_vector(phi, ql, d);

            const auto Jai_avg = 0.5 * (Jai_j + Jai_l);

            auto ul = phi_reader.get_value(ql);
            double d_jl = D(j, l);
            auto two_pt_flux = euler_CH_EC_flux<dim>(uj, ul, gas_gamma);
            flux_j -= 2.0 * d_jl * two_pt_flux * Jai_avg;
        }
        phi.submit_value((1.0 - alpha) * flux_j / Jdet_j, qj);
    }

    // Need to be careful to integrate for each flux dimension d,
    // otherwise the quadrature point values get overwritten for the next
    // dimension.
    phi.integrate_scatter(EvaluationFlags::values, dst);
}

template <int dim>
void FluidFluxESDGSEMOperator<dim>::calculate_first_order_ES_flux(
    LinearAlgebra::distributed::Vector<double> &dst,
    FEEvaluation<dim, -1, 0, dim + 2, double> &phi,
    const FEEvaluation<dim, -1, 0, dim + 2, double> &phi_reader,
    const std::vector<double> &quadrature_weights, const FullMatrix<double> &Q,
    unsigned int d, VectorizedArray<double> alpha, bool log) const {
    unsigned int fe_degree = discretization->get_fe_degree();
    unsigned int Np = fe_degree + 1;

    std::vector<Tensor<1, dim + 2, VectorizedArray<double>>> flux_differences(
        Np);

    unsigned int stride = pencil_stride(Np, d);
    for (unsigned int pencil_start : pencil_starts<dim>(Np, d)) {
        for (unsigned int i = 0; i < Np; i++) {
            for (unsigned int comp = 0; comp < dim + 2; comp++) {
                flux_differences[i][comp] = VectorizedArray(0.0);
            }
        }
        Tensor<1, dim, VectorizedArray<double>> n_i_iplus1 =
            scaled_contravariant_basis_vector(phi, pencil_start, d);

        const auto Jdet_0 = jacobian_determinant(phi, pencil_start);
        const auto f0 =
            euler_flux<dim>(phi_reader.get_value(pencil_start), gas_gamma) *
            n_i_iplus1;
        flux_differences[0] += alpha * f0 / quadrature_weights[0] / Jdet_0;

        // Index i runs over subcells, and at each subcell we consider the right
        // face. So should skip the final subcell whose right face is handled by
        // the usual numerical flux.
        for (unsigned int i = 0; i < Np - 1; i++) {
            unsigned int qi = pencil_start + stride * i;

            for (unsigned int m = 0; m < Np; m++) {
                unsigned int qm = pencil_start + stride * m;
                Tensor<1, dim, VectorizedArray<double>> Jad_m =
                    scaled_contravariant_basis_vector(phi, qm, d);
                n_i_iplus1 += Q(i, m) * Jad_m;
            }

            const auto Jdet_i = jacobian_determinant(phi, qi);
            const auto Jdet_i_plus_1 = jacobian_determinant(phi, qi + stride);

            const auto n_i_iplus1_norm = n_i_iplus1.norm();

            // We're going to calculate the numerical flux across the subcell
            // face from subcell i to subcell i+1.
            const auto left_state = phi_reader.get_value(qi);
            const auto right_state = phi_reader.get_value(qi + stride);
            const auto flux_dot_n =
                euler_CH_entropy_dissipating_flux<dim>(
                    left_state, right_state, n_i_iplus1 / n_i_iplus1_norm,
                    gas_gamma) *
                n_i_iplus1_norm;

            // Perform a face-centered flux difference, so subtract from the
            // left subcell and add to the right subcell. Equation (13) in
            // Henneman et al.
            flux_differences[i] +=
                (-alpha * flux_dot_n / quadrature_weights[i] / Jdet_i);
            flux_differences[i + 1] +=
                alpha * flux_dot_n / quadrature_weights[i + 1] / Jdet_i_plus_1;
        }

        // TODO: check that this subcell face normal vector is equal to
        // Jad_Np.
        for (unsigned int m = 0; m < Np; m++) {
            unsigned int qm = pencil_start + stride * m;
            Tensor<1, dim, VectorizedArray<double>> Jad_m =
                scaled_contravariant_basis_vector(phi, qm, d);
            n_i_iplus1 += Q(Np - 1, m) * Jad_m;
        }

        const auto qNp = pencil_start + stride * (Np - 1);
        const auto Jdet_Np = jacobian_determinant(phi, qNp);
        const auto fN =
            euler_flux<dim>(phi_reader.get_value(qNp), gas_gamma) * n_i_iplus1;
        flux_differences[Np - 1] +=
            (-alpha * fN / quadrature_weights[Np - 1] / Jdet_Np);

        if (log) {
        SHOW(d);
        }
        for (unsigned int i = 0; i < Np; i++) {
            if (log) {
                SHOW(i);
                SHOW(flux_differences[i]);
            }
            phi.submit_value(flux_differences[i], pencil_start + stride * i);
        }
    }

    phi.integrate_scatter(EvaluationFlags::values, dst);
}

template <int dim>
void FluidFluxESDGSEMOperator<dim>::local_apply_cell(
    const MatrixFree<dim, double> &mf,
    LinearAlgebra::distributed::Vector<double> &dst,
    const LinearAlgebra::distributed::Vector<double> &src,
    const std::pair<unsigned int, unsigned int> &cell_range) const {
    unsigned int fe_degree = discretization->get_fe_degree();
    unsigned int Np = fe_degree + 1;
    FEValues<dim> fe_values(discretization->get_mapping(),
                            discretization->get_fe(), QGaussLobatto<dim>(Np),
                            UpdateFlags::update_values);

    const std::vector<double> &quadrature_weights =
        fe_values.get_quadrature().get_weights();

    std::vector<unsigned int> n_coefs_per_dim = {};
    n_coefs_per_dim.push_back(Np);
    auto fe_collection =
        hp::FECollection<dim>(fe_values.get_fe().base_element(0));
    auto q_collection = hp::QCollection<dim>(fe_values.get_quadrature());
    FESeries::Legendre<dim> legendre(n_coefs_per_dim, fe_collection,
                                     q_collection, 0);

    FullMatrix<double> D(Np, Np);
    FullMatrix<double> Q(Np, Np);
    for (unsigned int j = 0; j < Np; j++) {
        for (unsigned int l = 0; l < Np; l++) {
            Point<dim> j_pt = fe_values.get_quadrature().point(j);
            D(j, l) = fe_values.get_fe().shape_grad(l, j_pt)[0];
            Q(j, l) = quadrature_weights[j] * D(j, l);
        }
    }

    for (unsigned int species_index = 0; species_index < n_species;
         species_index++) {
        unsigned int first_component = species_index * (dim + 2);
        FEEvaluation<dim, -1, 0, dim + 2, double> phi(mf, 0, 1,
                                                      first_component);
        FEEvaluation<dim, -1, 0, dim + 2, double> phi_reader(mf, 0, 1,
                                                             first_component);

        for (unsigned int cell = cell_range.first; cell < cell_range.second;
             ++cell) {
            phi.reinit(cell);
            phi_reader.reinit(cell);
            phi_reader.gather_evaluate(src, EvaluationFlags::values);
            phi_reader.read_dof_values(src);

            VectorizedArray<double> alpha =
                shock_indicators(phi_reader, legendre);
            //SHOW(alpha);

            for (unsigned int d = 0; d < dim; d++) {
                alpha = 1.0;
                //calculate_high_order_EC_flux(dst, phi, phi_reader, D, d, alpha, false);
                calculate_first_order_ES_flux(dst, phi, phi_reader,
                                              quadrature_weights, Q, d, alpha, cell==0);
            }
        }
    }
}

template <int dim>
void FluidFluxESDGSEMOperator<dim>::local_apply_face(
    const MatrixFree<dim, double> &mf,
    LinearAlgebra::distributed::Vector<double> &dst,
    const LinearAlgebra::distributed::Vector<double> &src,
    const std::pair<unsigned int, unsigned int> &face_range) const {
    for (unsigned int species_index = 0; species_index < n_species;
         species_index++) {
        unsigned int first_component = species_index * (dim + 2);
        FEFaceEvaluation<dim, -1, 0, dim + 2, double> phi_m(mf, true, 0, 1,
                                                            first_component);
        FEFaceEvaluation<dim, -1, 0, dim + 2, double> phi_p(mf, false, 0, 1,
                                                            first_component);

        for (unsigned int face = face_range.first; face < face_range.second;
             ++face) {
            phi_p.reinit(face);
            phi_p.gather_evaluate(src, EvaluationFlags::values);

            phi_m.reinit(face);
            phi_m.gather_evaluate(src, EvaluationFlags::values);

            for (const unsigned int q : phi_m.quadrature_point_indices()) {
                const auto n = phi_m.normal_vector(q);
                const auto flux_m =
                    euler_flux<dim>(phi_m.get_value(q), gas_gamma) * n;
                const auto flux_p =
                    euler_flux<dim>(phi_p.get_value(q), gas_gamma) * n;
                const auto numerical_flux =
                    euler_CH_entropy_dissipating_flux<dim>(
                        phi_m.get_value(q), phi_p.get_value(q),
                        phi_m.get_normal_vector(q), gas_gamma);

                phi_m.submit_value(flux_m - numerical_flux, q);
                phi_p.submit_value(numerical_flux - flux_p, q);
            }

            phi_m.integrate_scatter(EvaluationFlags::values, dst);
            phi_p.integrate_scatter(EvaluationFlags::values, dst);
        }
    }
}

template <int dim>
void FluidFluxESDGSEMOperator<dim>::local_apply_boundary_face(
    const MatrixFree<dim> &mf, LinearAlgebra::distributed::Vector<double> &dst,
    const LinearAlgebra::distributed::Vector<double> &src,
    const std::pair<unsigned int, unsigned int> &face_range,
    FiveMBoundaryIntegratedFluxesVector &d_dt_boundary_integrated_fluxes)
    const {
    for (unsigned int species_index = 0; species_index < n_species;
         species_index++) {
        EulerBCMap<dim> &bc_map = species.at(species_index)->bc_map;
        unsigned int first_component = species_index * (dim + 2);
        FEFaceEvaluation<dim, -1, 0, dim + 2, double> phi(mf, true, 0, 0,
                                                          first_component);
        FEFaceEvaluation<dim, -1, 0, dim + 2, double>
            phi_boundary_flux_integrator(mf, true, 0, 0, first_component);

        for (unsigned int face = face_range.first; face < face_range.second;
             ++face) {
            phi.reinit(face);
            phi.gather_evaluate(src, EvaluationFlags::values);
            phi_boundary_flux_integrator.reinit(face);

            const auto boundary_id = mf.get_boundary_id(face);

            for (const unsigned int q : phi.quadrature_point_indices()) {
                const Tensor<1, dim + 2, VectorizedArray<double>> w_m =
                    phi.get_value(q);
                const Tensor<1, dim, VectorizedArray<double>> normal =
                    phi.normal_vector(q);

                auto rho_u_dot_n = w_m[1] * normal[0];
                for (unsigned int d = 1; d < dim; d++) {
                    rho_u_dot_n += w_m[1 + d] * normal[d];
                }

                // bool at_outflow = false;
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
                    // at_outflow = true;
                } else if (bc_map.is_supersonic_outflow(boundary_id)) {
                    w_p = w_m;
                    // at_outflow = true;
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

                auto analytic_flux = euler_flux<dim>(w_m, gas_gamma) * normal;
                auto numerical_flux =
                    euler_numerical_flux<dim>(w_m, w_p, normal, gas_gamma);

                phi.submit_value(analytic_flux - numerical_flux, q);
                phi_boundary_flux_integrator.submit_value(numerical_flux, q);
            }
            phi.integrate_scatter(EvaluationFlags::values, dst);

            /**
             * While we are here at this face, integrate the numerical flux
             * across it for use in diagnostics.
             */
            Tensor<1, dim + 2, VectorizedArray<double>>
                integrated_boundary_flux =
                    phi_boundary_flux_integrator.integrate_value();
            for (unsigned int lane = 0;
                 lane < mf.n_active_entries_per_face_batch(face); lane++) {
                Tensor<1, dim + 2, double> tensor;
                for (unsigned int comp = 0; comp < dim + 2; comp++) {
                    tensor[comp] = integrated_boundary_flux[comp][lane];
                }
                d_dt_boundary_integrated_fluxes.add<dim>(boundary_id, tensor);
            }
        }
    }
}

template <int dim>
double FluidFluxESDGSEMOperator<dim>::recommend_dt(
    const MatrixFree<dim, double> &mf, const FiveMSolutionVec &sol) {
    double max_transport_speed = compute_cell_transport_speed(mf, sol.mesh_sol);
    unsigned int fe_degree = discretization->get_fe_degree();
    return 0.5 / (max_transport_speed * (fe_degree + 1) * (fe_degree + 1));
}

template <int dim>
double FluidFluxESDGSEMOperator<dim>::compute_cell_transport_speed(
    const MatrixFree<dim, double> &mf,
    const LinearAlgebra::distributed::Vector<double> &solution) const {
    using VA = VectorizedArray<Number>;

    Number max_transport = 0;

    for (unsigned int species_index = 0; species_index < n_species;
         species_index++) {
        unsigned int first_component = species_index * (dim + 2);

        FEEvaluation<dim, -1, 0, dim + 2, Number> phi(mf, 0, 1,
                                                      first_component);

        for (unsigned int cell = 0; cell < mf.n_cell_batches(); ++cell) {
            phi.reinit(cell);
            phi.gather_evaluate(solution, EvaluationFlags::values);
            VA local_max = 0.;
            for (const unsigned int q : phi.quadrature_point_indices()) {
                const auto solution = phi.get_value(q);
                const auto velocity = euler_velocity<dim>(solution);
                const auto pressure = euler_pressure<dim>(solution, gas_gamma);

                const auto inverse_jacobian = phi.inverse_jacobian(q);
                const auto convective_speed = inverse_jacobian * velocity;
                VA convective_limit = 0.;
                for (unsigned int d = 0; d < dim; ++d)
                    convective_limit = std::max(convective_limit,
                                                std::abs(convective_speed[d]));

                const auto speed_of_sound =
                    std::sqrt(gas_gamma * pressure * (1. / solution[0]));

                Tensor<1, dim, VA> eigenvector;
                for (unsigned int d = 0; d < dim; ++d) eigenvector[d] = 1.;
                for (unsigned int i = 0; i < 5; ++i) {
                    eigenvector = transpose(inverse_jacobian) *
                                  (inverse_jacobian * eigenvector);
                    VA eigenvector_norm = 0.;
                    for (unsigned int d = 0; d < dim; ++d)
                        eigenvector_norm = std::max(eigenvector_norm,
                                                    std::abs(eigenvector[d]));
                    eigenvector /= eigenvector_norm;
                }
                const auto jac_times_ev = inverse_jacobian * eigenvector;
                const auto max_eigenvalue =
                    std::sqrt((jac_times_ev * jac_times_ev) /
                              (eigenvector * eigenvector));
                local_max =
                    std::max(local_max, max_eigenvalue * speed_of_sound +
                                            convective_limit);
            }

            for (unsigned int v = 0;
                 v < mf.n_active_entries_per_cell_batch(cell); ++v) {
                for (unsigned int d = 0; d < 3; ++d)
                    max_transport = std::max(max_transport, local_max[v]);
            }
        }
    }
    max_transport = Utilities::MPI::max(max_transport, MPI_COMM_WORLD);

    return max_transport;
}

template <int dim>
VectorizedArray<double> FluidFluxESDGSEMOperator<dim>::shock_indicators(
    const FEEvaluation<dim, -1, 0, dim + 2, double> &phi,
    FESeries::Legendre<dim> &legendre) const {
    unsigned int Np = discretization->get_fe_degree() + 1;
    AssertThrow(Np >= 2, ExcMessage("Shock indicators are not supported nor "
                                    "needed for P0 polynomial bases"));

    VectorizedArray<double> alphas;

    Vector<double> p_times_rho;
    p_times_rho.reinit(phi.dofs_per_component);
    for (unsigned int lane = 0; lane < VectorizedArray<double>::size();
         lane++) {
        for (unsigned int dof = 0; dof < phi.dofs_per_component; dof++) {
            const auto q_dof = phi.get_dof_value(dof);
            const auto rho = q_dof[0][lane];
            const auto p = euler_pressure<dim>(q_dof, gas_gamma)[lane];
            p_times_rho(dof) = p * rho;
        }

        TableIndices<dim> sizes;
        for (unsigned int d = 0; d < dim; d++) {
            sizes[d] = Np;
        }
        Table<dim, double> legendre_coefs;
        legendre_coefs.reinit(sizes);
        legendre.calculate(p_times_rho, 0, legendre_coefs);

        /**
         * It is unclear from Hennemann et al. how to cut off the modal energy
         * for higher dimensional functions.
         *
         * Based on what Trixi.jl does, we apply a cutoff to the /maximum/
         * single variable degree of a mode. So rather than cutting off the tip
         * of the modal cube, we cut off the whole degree-N shell.
         */
        const std::function<std::pair<bool, unsigned int>(
            const TableIndices<dim> &index)>
            group_leading_coefs = [&Np](const TableIndices<dim> &index)
            -> std::pair<bool, unsigned int> {
            std::size_t max_degree = 0;
            for (unsigned int d = 0; d < dim; d++) {
                max_degree = std::max(index[d], max_degree);
            }
            if (max_degree < Np - 1) {
                return std::make_pair(true, 0);
            } else if (max_degree == Np - 1) {
                return std::make_pair(true, 1);
            } else if (max_degree == Np) {
                return std::make_pair(true, 2);
            } else {
                // This should be impossible
                Assert(false, ExcMessage("Unreachable"));
                return std::make_pair(false, 3);
            }
        };
        const std::pair<std::vector<unsigned int>, std::vector<double>>
            grouped_coefs = FESeries::process_coefficients(
                legendre_coefs, group_leading_coefs,
                VectorTools::NormType::L2_norm);
        double total_energy = 0.;
        double total_energy_minus_1 = 0.;
        double top_mode = 0.;
        double top_mode_minus_1 = 0.;
        for (unsigned int i = 0; i < grouped_coefs.first.size(); i++) {
            unsigned int predicate = grouped_coefs.first[i];
            double sqrt_energy = grouped_coefs.second[i];
            double energy = sqrt_energy * sqrt_energy;
            if (predicate == 0) {
                total_energy += energy;
                total_energy_minus_1 += energy;
            } else if (predicate == 1) {
                top_mode_minus_1 += energy;
                total_energy_minus_1 += energy;
                total_energy += energy;
            } else if (predicate == 2) {
                top_mode += energy;
                total_energy += energy;
            }
        }
        double E = std::max(top_mode / total_energy,
                            top_mode_minus_1 / total_energy_minus_1);
        double T = 0.5 * std::pow(10.0, -1.8 * std::pow(Np, 0.25));
        double s = 9.21024;
        double alpha = 1.0 / (1.0 + std::exp(-s / T * (E - T)));
        double alpha_max = 1.0;

        if (alpha < 1e-3) {
            alpha = 0.0;
        } else if (alpha > alpha_max) {
            alpha = alpha_max;
        }
        alphas[lane] = alpha;
    }
    return alphas;
}

}  // namespace five_moment
}  // namespace warpii
