#include <deal.II/base/point.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

using namespace dealii;

  template <int dim, typename Number>
VectorizedArray<Number>
evaluate_function(const Function<dim>                       &function,
                const Point<dim, VectorizedArray<Number>> &p_vectorized,
                const unsigned int                         component)
{
VectorizedArray<Number> result;
for (unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v)
  {
    Point<dim> p;
    for (unsigned int d = 0; d < dim; ++d)
      p[d] = p_vectorized[d][v];
    result[v] = function.value(p, component);
  }
return result;
}


template <int dim, typename Number, int n_components = dim + 2>
Tensor<1, n_components, VectorizedArray<Number>>
evaluate_function(const Function<dim>                       &function,
                const Point<dim, VectorizedArray<Number>> &p_vectorized)
{
AssertDimension(function.n_components, n_components);
Tensor<1, n_components, VectorizedArray<Number>> result;
for (unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v)
  {
    Point<dim> p;
    for (unsigned int d = 0; d < dim; ++d)
      p[d] = p_vectorized[d][v];
    for (unsigned int d = 0; d < n_components; ++d)
      result[d][v] = function.value(p, d);
  }
return result;
}

