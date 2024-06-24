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
#include "../dg_solution_helper.h"
#include "../../dgsem/nodal_dg_discretization.h"
#include "../euler.h"
#include "../solution_vec.h"
#include "../species.h"
#include "jacobian_utils.h"

namespace warpii {
namespace five_moment {

template <int dim>
class SplitFormVolumeFlux {
   public:
    SplitFormVolumeFlux(
        std::shared_ptr<NodalDGDiscretization<dim>> discretization,
        double gas_gamma)
        : Np(discretization->get_fe_degree()+1), gas_gamma(gas_gamma), D(Np, Np) {

        FEValues<dim> fe_values(
            discretization->get_mapping(), discretization->get_fe(),
            QGaussLobatto<dim>(Np), UpdateFlags::update_values);

        for (unsigned int j = 0; j < Np; j++) {
            for (unsigned int l = 0; l < Np; l++) {
                Point<dim> j_pt = fe_values.get_quadrature().point(j);
                D(j, l) = fe_values.get_fe().shape_grad(l, j_pt)[0];
            }
        }
    }

    void calculate_flux(
        LinearAlgebra::distributed::Vector<double> &dst,
        FEEvaluation<dim, -1, 0, 5, double> &phi,
        const FEEvaluation<dim, -1, 0, 5, double> &phi_reader,
        VectorizedArray<double> alpha,
        bool log) const;

   private:
    unsigned int Np;
    double gas_gamma;
    FullMatrix<double> D;
};

template <int dim>
void SplitFormVolumeFlux<dim>::calculate_flux(
    LinearAlgebra::distributed::Vector<double> &dst,
    FEEvaluation<dim, -1, 0, 5, double> &phi,
    const FEEvaluation<dim, -1, 0, 5, double> &phi_reader,
    VectorizedArray<double> alpha,
    bool /* log */) const {
    for (unsigned int d = 0; d < dim; d++) {
        for (const unsigned int qj : phi.quadrature_point_indices()) {
            auto Jdet_j = jacobian_determinant(phi, qj);
            auto Jai_j = scaled_contravariant_basis_vector(phi, qj, d);

            unsigned int j = quad_point_1d_index<dim>(qj, Np, d);
            auto uj = phi_reader.get_value(qj);
            Tensor<1, 5, VectorizedArray<double>> flux_j;
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
}

}  // namespace five_moment
}  // namespace warpii
