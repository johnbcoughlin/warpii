#include "dof_utils.h"
#include <deal.II/base/exceptions.h>

namespace warpii {
    using namespace dealii;

unsigned int pencil_stride(unsigned int Np, unsigned int d) {
    if (d == 0) {
        return 1;
    } else if (d == 1) {
        return Np;
    } else if (d == 2) {
        return Np * Np;
    } else {
        Assert(false, ExcMessage("n_dims appears to be 4 or more"));
        return -1;
    }
}

template <>
unsigned int pencil_base<1>(const unsigned int, const unsigned int , const unsigned int ) {
    return 0;
}

template <>
unsigned int pencil_base<2>(const unsigned int q, const unsigned int Np, const unsigned int d) {
    Assert(q < Np * Np, ExcMessage("Quadrature point index was too large"));
    if (d == 0) {
        // It's an x pencil
        unsigned int k = 0;
        unsigned int i = q;
        while (i >= Np) {
            i -= Np;
            k++;
        }
        return k * Np;
    } else {
        // It's a y pencil
        unsigned int i = q;
        while (i >= Np) {
            i -= Np;
        }
        return i;
    }
}

template <>
unsigned int quadrature_point_neighbor<1>(const unsigned int,
        const unsigned int k, unsigned int, unsigned int) {
    return k;
}

template <>
unsigned int quadrature_point_neighbor<2>(const unsigned int q,
        const unsigned int k, const unsigned int Np, const unsigned int d) {
    unsigned int stride = pencil_stride(Np, d);
    return pencil_base<2>(q, Np, d) + stride * k;
}

template <>
unsigned int quad_point_1d_index<1>(const unsigned int q, const unsigned int, const unsigned int) {
    return q;
}

template <>
unsigned int quad_point_1d_index<2>(const unsigned int q, const unsigned int Np, const unsigned int d) {
    unsigned int stride = pencil_stride(Np, d);
    unsigned int i = pencil_base<2>(q, Np, d);
    unsigned int c = 0;
    while (i < q) {
        i += stride;
        c++;
    }
    return c;
}

}
