#pragma once
#include <deal.II/base/config.h>
#include <deal.II/base/table_indices.h>
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

namespace warpii {
template <int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE Tensor<1, dim, Number> tensor_row(
        const Tensor<2, dim, Number> &matrix,
        unsigned int row) {
    Tensor<1, dim, Number> result;
    for (unsigned int d = 0; d < dim; ++d)
        result[d] = matrix[TableIndices<2>(row, d)];
    return result;
}

template <int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE Tensor<1, dim, Number> tensor_column(
        const Tensor<2, dim, Number> &matrix,
        unsigned int col) {
    Tensor<1, dim, Number> result;
    for (unsigned int d = 0; d < dim; ++d)
        result[d] = matrix[TableIndices<2>(d, col)];
    return result;
}

template <typename Number>
inline DEAL_II_ALWAYS_INLINE Number determinant(const Tensor<2, 1, Number> &matrix) {
    return matrix[TableIndices<2>(0, 0)];
}

template <typename Number>
inline DEAL_II_ALWAYS_INLINE Number determinant(const Tensor<2, 2, Number> &matrix) {
    const auto a = matrix[TableIndices<2>(0, 0)];
    const auto b = matrix[TableIndices<2>(0, 1)];
    const auto c = matrix[TableIndices<2>(1, 0)];
    const auto d = matrix[TableIndices<2>(1, 1)];
    return a * d - b * c;
}
}
