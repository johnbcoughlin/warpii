#pragma once
#include <deal.II/base/vectorization.h>
#include <deal.II/fe/fe_series.h>
#include <deal.II/matrix_free/fe_evaluation.h>

#include "nodal_dg_discretization.h"

using namespace dealii;

namespace warpii {

template <int dim>
FESeries::Legendre<dim> initialize_legendre(
    NodalDGDiscretization<dim> &discretization) {
    unsigned int fe_degree = discretization.get_fe_degree();
    unsigned int Np = fe_degree + 1;
    std::vector<unsigned int> n_coefs_per_dim = {};
    n_coefs_per_dim.push_back(Np);
    FESeries::Legendre<dim> legendre(
        n_coefs_per_dim, discretization.get_dummy_fe_collection(),
        discretization.get_dummy_q_collection(), 0);
    return legendre;
}

/**
 * Computes the troubled cell indicator of Persson and Peraire,
 * "Sub-Cell Shock Capturing for Discontinuous Galerkin Methods"
 */
template <int dim>
class PerssonPeraireShockIndicator {
   public:
    PerssonPeraireShockIndicator(
        std::shared_ptr<NodalDGDiscretization<dim>> discretization)
        : Np(discretization->get_fe_degree() + 1),
          legendre(initialize_legendre(*discretization)) {}

    double compute_shock_indicator(const Vector<double> &u);

   private:
    unsigned int Np;
    FESeries::Legendre<dim> legendre;
};

template <int dim>
double PerssonPeraireShockIndicator<dim>::compute_shock_indicator(
    const Vector<double> &u) {
    AssertThrow(Np >= 2, ExcMessage("Shock indicators are not supported nor "
                                    "needed for P0 polynomial bases"));

    TableIndices<dim> sizes;
    for (unsigned int d = 0; d < dim; d++) {
        sizes[d] = Np;
    }
    Table<dim, double> legendre_coefs;
    legendre_coefs.reinit(sizes);
    legendre.calculate(u, 0, legendre_coefs);

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
        group_leading_coefs = [&](const TableIndices<dim> &index)
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
        grouped_coefs =
            FESeries::process_coefficients(legendre_coefs, group_leading_coefs,
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
    double alpha_max = 0.5;

    if (alpha < 1e-3) {
        alpha = 0.0;
    } else if (alpha > alpha_max) {
        alpha = alpha_max;
    }
    return alpha;
}

}  // namespace warpii
