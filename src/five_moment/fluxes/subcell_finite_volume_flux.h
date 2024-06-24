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

#include "../../dof_utils.h"
#include "../../utilities.h"
#include "../euler.h"
#include "../solution_vec.h"
#include "../species.h"
#include "jacobian_utils.h"
#include "../../dgsem/nodal_dg_discretization.h"

namespace warpii {
namespace five_moment {

template <int dim>
class SubcellFiniteVolumeFlux {
   public:
    SubcellFiniteVolumeFlux(
        NodalDGDiscretization<dim>& discretization,
        double gas_gamma)
        : Np(discretization.get_fe_degree() + 1),
          gas_gamma(gas_gamma),
          Q(Np, Np) {
        FEValues<dim> fe_values(
            discretization.get_mapping(), discretization.get_fe(),
            QGaussLobatto<dim>(Np), UpdateFlags::update_values);

        Quadrature<1> quad = QGaussLobatto<1>(Np);
        for (unsigned int i = 0; i < Np; i++) {
            quadrature_weights.push_back(quad.get_weights().at(i));
        }

        FullMatrix<double> D(Np, Np);
        for (unsigned int j = 0; j < Np; j++) {
            for (unsigned int l = 0; l < Np; l++) {
                Point<dim> j_pt = fe_values.get_quadrature().point(j);
                D(j, l) = fe_values.get_fe().shape_grad(l, j_pt)[0];
                Q(j, l) = quadrature_weights[j] * D(j, l);
            }
        }
    }

    void calculate_flux(
        LinearAlgebra::distributed::Vector<double> &dst,
        FEEvaluation<dim, -1, 0, 5, double> &phi,
        const FEEvaluation<dim, -1, 0, 5, double> &phi_reader,
        VectorizedArray<double> alpha, bool log) const;

   private:
    unsigned int Np;
    double gas_gamma;
    std::vector<double> quadrature_weights;
    FullMatrix<double> Q;
};

template <int dim>
void SubcellFiniteVolumeFlux<dim>::calculate_flux(
    LinearAlgebra::distributed::Vector<double> &dst,
    FEEvaluation<dim, -1, 0, 5, double> &phi,
    const FEEvaluation<dim, -1, 0, 5, double> &phi_reader,
    VectorizedArray<double> alpha, bool /* log */) const {

    for (unsigned int d = 0; d < dim; d++) {
        std::vector<Tensor<1, 5, VectorizedArray<double>>>
            flux_differences(Np);

        unsigned int stride = pencil_stride(Np, d);
        for (unsigned int pencil_start : pencil_starts<dim>(Np, d)) {
            for (unsigned int i = 0; i < Np; i++) {
                for (unsigned int comp = 0; comp < 5; comp++) {
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

            // Index i runs over subcells, and at each subcell we consider the
            // right face. So should skip the final subcell whose right face is
            // handled by the usual numerical flux.
            for (unsigned int i = 0; i < Np - 1; i++) {
                unsigned int qi = pencil_start + stride * i;

                for (unsigned int m = 0; m < Np; m++) {
                    unsigned int qm = pencil_start + stride * m;
                    Tensor<1, dim, VectorizedArray<double>> Jad_m =
                        scaled_contravariant_basis_vector(phi, qm, d);
                    n_i_iplus1 += Q(i, m) * Jad_m;
                }

                const auto Jdet_i = jacobian_determinant(phi, qi);
                const auto Jdet_i_plus_1 =
                    jacobian_determinant(phi, qi + stride);

                const auto n_i_iplus1_norm = n_i_iplus1.norm();

                // We're going to calculate the numerical flux across the
                // subcell face from subcell i to subcell i+1.
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
                flux_differences[i + 1] += alpha * flux_dot_n /
                                           quadrature_weights[i + 1] /
                                           Jdet_i_plus_1;
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
                euler_flux<dim>(phi_reader.get_value(qNp), gas_gamma) *
                n_i_iplus1;
            flux_differences[Np - 1] +=
                (-alpha * fN / quadrature_weights[Np - 1] / Jdet_Np);

            for (unsigned int i = 0; i < Np; i++) {
                phi.submit_value(flux_differences[i],
                                 pencil_start + stride * i);
            }
        }

        phi.integrate_scatter(EvaluationFlags::values, dst);
    }
}

}  // namespace five_moment
}  // namespace warpii
