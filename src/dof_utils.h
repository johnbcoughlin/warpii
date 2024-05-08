#pragma once

namespace warpii {
unsigned int quadrature_point_neighbor(
        unsigned int, unsigned int k, unsigned int, unsigned int d);

/**
 * Transform from q = (i, j, k) to one of i, j, or k.
 * q must be an index of a tensor-product array
 */
unsigned int quad_point_1d_index(unsigned int q, unsigned int Np, 
        unsigned int d);
}
