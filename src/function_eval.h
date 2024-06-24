#pragma once
#include <deal.II/base/point.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

using namespace dealii;

using Number = double;

  template <int dim, typename Number, typename VectorizedNumber>
VectorizedNumber
evaluate_function(const Function<dim>                       &function,
                const Point<dim, VectorizedNumber> &p_vectorized,
                const unsigned int                         component)
{
VectorizedNumber result;
for (unsigned int v = 0; v < VectorizedNumber::size(); ++v)
  {
    Point<dim> p;
    for (unsigned int d = 0; d < dim; ++d)
      p[d] = p_vectorized[d][v];
    result[v] = function.value(p, component);
  }
return result;
}


template <int dim, typename Number, int n_components, typename VectorizedNumber>
Tensor<1, n_components, VectorizedNumber>
evaluate_function(const Function<dim>                       &function,
                const Point<dim, VectorizedNumber> &p_vectorized)
{
AssertDimension(function.n_components, n_components);
Tensor<1, n_components, VectorizedNumber> result;
for (unsigned int v = 0; v < VectorizedNumber::size(); ++v)
  {
    Point<dim> p;
    for (unsigned int d = 0; d < dim; ++d)
      p[d] = p_vectorized[d][v];
    for (unsigned int d = 0; d < n_components; ++d)
      result[d][v] = function.value(p, d);
  }
return result;
}

