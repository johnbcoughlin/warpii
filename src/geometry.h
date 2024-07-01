#include <deal.II/base/tensor.h>

using namespace dealii;

namespace warpii {

template <int dim, typename Number>
Tensor<1, 3, Number> at_least_3d(Tensor<1, dim, Number> n) {
    Tensor<1, 3, Number> result;
    for (unsigned int d = 0; d < dim; d++) {
        result[d] = n[d];
    }
    for (unsigned int d = dim; d < 3; d++) {
        result[d] = 0.0;
    }
    return result;
}

/**
 * Given a normal vector to a plane, computes two basis vectors for that plane,
 * which we call the tangent and binormal vectors.
 */
template <typename Number>
std::pair<Tensor<1, 3, Number>, Tensor<1, 3, Number>> tangent_and_binormal(
    const Tensor<1, 3, Number> n) {
    // Construct a reference vector pointing in an arbitrary direction
    Tensor<1, 3, Number> a;
    a[0] = Number(0.);
    a[1] = Number(0.);
    a[2] = Number(1.);
    for (unsigned int v = 0; v < Number::size(); v++) {
        // If n is too nearly colinear with a, pick a different a.
        // By the Brouwer fixed point theorem, no continuous mapping from S^2 to itself
        // can avoid having a fixed point. This introduced a discontinuity if it detects
        // that n is at the fixed point.
        if (std::abs(a * n) >= 0.9) {
            a[1] = Number(1.);
            a[2] = Number(0.);
        }
    }

    auto tangent = a - (a * n) * n;
    tangent = tangent / tangent.norm();

    auto binormal = cross_product_3d(n, tangent);
    return std::make_pair(tangent, binormal);
}

}  // namespace warpii
