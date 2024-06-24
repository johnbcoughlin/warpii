#pragma once

#include <deal.II/base/exceptions.h>
#include <vector>

using namespace dealii;

namespace warpii {

/**
 * This function is based on the documentation for the ordering of FE_DGQ
 * DOFs, found here:
 * https://www.dealii.org/current/doxygen/deal.II/classFE__DGQ.html
 */
template <int dim>
std::vector<unsigned int> pencil_starts(unsigned int Np, unsigned int q);

/**
 * This function is based on the documentation for the ordering of FE_DGQ
 * DOFs, found here:
 * https://www.dealii.org/current/doxygen/deal.II/classFE__DGQ.html
 */
unsigned int pencil_stride(unsigned int Np, unsigned int d);

template <int dim>
unsigned int pencil_base(const unsigned int q, const unsigned int Np, const unsigned int d);

/**
 * Params are
 * (q, k, Np, dim)
 */
template <int dim>
unsigned int quadrature_point_neighbor(unsigned int, unsigned int k, unsigned int /*Np*/, unsigned int);

/**
 * Params are
 * (q, k, dim)
 *
 * Transform from q = (i, j, k) to one of i, j, or k.
 * q must be an index of a tensor-product array
 */
template <int dim>
unsigned int quad_point_1d_index(unsigned int q, unsigned int , unsigned int);

enum ZeroOutPolicy {
    DO_ZERO_DST_VECTOR,
    DO_NOT_ZERO_DST_VECTOR,
};

}
