#pragma once

#include <deal.II/base/array_view.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/fe/fe_values_extractors.h>
#include <deal.II/lac/vector.h>

#include "dg_discretization.h"
#include "euler.h"
#include "species.h"
#include "../dof_utils.h"

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
        LinearAlgebra::distributed::Vector<double> &dst,
        const LinearAlgebra::distributed::Vector<double> &u,
        std::vector<LinearAlgebra::distributed::Vector<double>> &sol_registers,
        const double dt, const double t, const double alpha = 1.0,
        const double beta = 0.0) const;

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

    std::shared_ptr<FiveMomentDGDiscretization<dim>> discretization;
    double gas_gamma;
    unsigned int n_species;
    std::vector<std::shared_ptr<Species<dim>>> species;
};

template <int dim>
void FluidFluxESDGSEMOperator<dim>::perform_forward_euler_step(
    LinearAlgebra::distributed::Vector<double> &dst,
    const LinearAlgebra::distributed::Vector<double> &u,
    std::vector<LinearAlgebra::distributed::Vector<double>> &sol_registers, 
    const double dt,
    const double t, 
    const double alpha, const double beta) const {
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

        discretization->mf.loop(
            &FluidFluxESDGSEMOperator<dim>::local_apply_cell,
            &FluidFluxESDGSEMOperator<dim>::local_apply_face,
            &FluidFluxESDGSEMOperator<dim>::local_apply_boundary_face, this,
            Mdudt_register, u, true,
            MatrixFree<dim, double>::DataAccessOnFaces::values,
            MatrixFree<dim, double>::DataAccessOnFaces::values);
    }

    {
        discretization->mf.cell_loop(
            &FluidFluxESDGSEMOperator<dim>::local_apply_inverse_mass_matrix, this,
            dudt_register, Mdudt_register,
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
        // discretization->mf.cell_loop(
        //&FluidFluxDGOperator<dim>::local_apply_positivity_limiter, this,
        // dst, sol_before_limiting);
        dst.sadd(0.0, 1.0, sol_before_limiting);
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
    const FEEvaluation<dim, -1, 0, dim + 2, double>& phi, unsigned int q) {
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
    const FEEvaluation<dim, -1, 0, dim + 2, double>& phi, 
    unsigned int q, unsigned int d) {
    Tensor<2, dim, VectorizedArray<double>> Jinv_j = phi.inverse_jacobian(q);
    VectorizedArray<double> Jdet_j = jacobian_determinant(phi, q);

    return Jdet_j * tensor_column(Jinv_j, d);
}

template <int dim>
void FluidFluxESDGSEMOperator<dim>::local_apply_cell(
    const MatrixFree<dim, double> &mf,
    LinearAlgebra::distributed::Vector<double> &dst,
    const LinearAlgebra::distributed::Vector<double> &src,
    const std::pair<unsigned int, unsigned int> &cell_range) const {

    unsigned int fe_degree = discretization->get_fe_degree();
    unsigned int Np = fe_degree + 1;
    FEValues<dim> fe_values(discretization->get_mapping(), discretization->get_fe(),
            QGaussLobatto<dim>(Np), UpdateFlags::update_values);

    FullMatrix<double> D(Np, Np);
    for (unsigned int j = 0; j < Np; j++) {
        for (unsigned int l = 0; l < Np; l++) {
            Point<dim> j_pt = fe_values.get_quadrature().point(j);
            D(j, l) = fe_values.get_fe().shape_grad(l, j_pt)[0];
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

            for (unsigned int d = 0; d < dim; d++) {
                Tensor<1, dim, VectorizedArray<double>> unit_basis_vec;
                for (unsigned int di = 0; di < dim; di++) {
                    unit_basis_vec[di] = VectorizedArray((di == d) ? 1.0 : 0.0);
                }

                for (const unsigned int qj : phi.quadrature_point_indices()) {
                    auto Jdet_j = jacobian_determinant(phi, qj);
                    auto Jai_j = scaled_contravariant_basis_vector(phi, qj, d);

                    unsigned int j = quad_point_1d_index(qj, Np, d);
                    auto uj = phi_reader.get_value(qj);
                    Tensor<1, dim+2, VectorizedArray<double>> flux_j;
                    for (unsigned int di = 0; di < dim; di++) {
                        flux_j[di] = 0.0;
                    }

                    for (unsigned int l = 0; l < Np; l++) {
                        unsigned int ql = quadrature_point_neighbor(qj, l, Np, d);
                        Tensor<2, dim, VectorizedArray<double>> Jinv_l = phi.inverse_jacobian(ql);
                        VectorizedArray<double> Jdet_l = 1.0 / warpii::determinant(Jinv_l);
                        auto Jai_l = Jdet_l * tensor_column(Jinv_l, d);

                        const auto Jai_avg = 0.5 * (Jai_j + Jai_l);

                        auto ul = phi_reader.get_value(ql);
                        double d_jl = D(j, l);
                        auto two_pt_flux = euler_central_flux<dim>(uj, ul, gas_gamma);
                        flux_j -= 2.0 * d_jl * two_pt_flux * Jai_avg;
                    }
                    phi.submit_value(flux_j / Jdet_j, qj);
                }

                // Need to be careful to integrate after each flux dimension d,
                // otherwise the quadrature point values get overwritten.
                phi.integrate_scatter(EvaluationFlags::values, dst);
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
                const auto flux_m = euler_flux<dim>(phi_m.get_value(q), gas_gamma) * n;
                const auto flux_p = euler_flux<dim>(phi_p.get_value(q), gas_gamma) * n;
                const auto numerical_flux = euler_numerical_flux<dim>(
                    phi_m.get_value(q), phi_p.get_value(q),
                    phi_m.normal_vector(q), gas_gamma);

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

                auto analytic_flux = euler_flux<dim>(w_m, gas_gamma) * normal;
                auto numerical_flux =
                    euler_numerical_flux<dim>(w_m, w_p, normal, gas_gamma);

                if (at_outflow) {
                    for (unsigned int v = 0;
                         v < VectorizedArray<double>::size(); ++v) {
                        if (rho_u_dot_n[v] < -1e-12) {
                            for (unsigned int d = 0; d < 1; ++d) {
                                numerical_flux[d + 1][v] = 0.;
                            }
                        }
                    }
                }

                phi.submit_value(analytic_flux - numerical_flux, q);
            }
            phi.integrate_scatter(EvaluationFlags::values, dst);
        }
    }
}


}  // namespace five_moment
}  // namespace warpii
