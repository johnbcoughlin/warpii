#include "dof_utils.h"
#include <deal.II/base/exceptions.h>

namespace warpii {
    using namespace dealii;

/**
 * Params are
 * (q, k, Np, dim)
 */
unsigned int quadrature_point_neighbor(unsigned int, unsigned int k, unsigned int, unsigned int d) {
    AssertThrow(d == 0, ExcMessage("We only know how to find quadrature point neighbors for dim = 1"));
    return k;
}

/**
 * Params are
 * (q, k, dim)
 */
unsigned int quad_point_1d_index(unsigned int q, unsigned int , unsigned int d) {
    AssertThrow(d == 0, ExcMessage("We only know how to find quadrature point neighbors for dim = 1"));
    return q;
}

}
