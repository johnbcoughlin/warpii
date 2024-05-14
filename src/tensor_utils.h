#pragma once
#include <deal.II/base/config.h>
#include <deal.II/base/table_indices.h>
#include <deal.II/base/tensor.h>
#include <limits>

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

template <int n_components, typename Number>
inline DEAL_II_ALWAYS_INLINE Tensor<1, n_components, Number> 
componentwise_product(
        const Tensor<1, n_components, Number> &a,
        const Tensor<1, n_components, Number> &b) {
    Tensor<1, n_components, Number> result;
    for (unsigned int i = 0; i < n_components; i++) {
        result[i] = a[i] * b[i];
    }
    return result;
}

template <int n_components, typename Number>
inline DEAL_II_ALWAYS_INLINE Number sum(const Tensor<1, n_components, Number> &a) {
    Number result(0.0);
    for (unsigned int i = 0; i < n_components; i++) {
        result += a[i];
    }
    return result;
}

template <int n_components, typename Number>
inline DEAL_II_ALWAYS_INLINE Number max_abs(const Tensor<1, n_components, Number> &a) {
    Number result(0.0);
    for (unsigned int i = 0; i < n_components; i++) {
        result = std::max(std::abs(a[i]), result);
    }
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
