#pragma once

#include <deal.II/base/tensor.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include "../../tensor_utils.h"

using namespace dealii;

namespace warpii {
    namespace five_moment {

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

    }
}
