#include <deal.II/base/tensor.h>

using namespace dealii;

template <int n_components, int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE Tensor<1, n_components, Number> operator*(
    const Tensor<1, n_components, Tensor<1, dim, Number>> &matrix,
    const Tensor<1, dim, Number> &vector) {
    Tensor<1, n_components, Number> result;
    for (unsigned int d = 0; d < n_components; ++d)
        result[d] = matrix[d] * vector;
    return result;
}

