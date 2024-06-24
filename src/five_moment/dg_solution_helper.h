#pragma once

#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/base/tensor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/numerics/vector_tools_common.h>
#include <deal.II/numerics/vector_tools_integrate_difference.h>
#include <deal.II/numerics/data_out.h>

#include "../function_eval.h"
#include "../dgsem/nodal_dg_discretization.h"
#include "../grid.h"

namespace warpii {
namespace five_moment {
template <int dim>
class FiveMomentDGSolutionHelper {
   public:
    FiveMomentDGSolutionHelper(std::shared_ptr<NodalDGDiscretization<dim>> discretization):
        discretization(discretization) {}

    void project_fluid_quantities(
        const Function<dim> &function,
        LinearAlgebra::distributed::Vector<double> &solution,
        unsigned int species_index) const;

    /**
     * Compute the L^2 norm of the difference between `solution` and `f`,
     * integrated over the domain.
     */
    double compute_global_error(
        LinearAlgebra::distributed::Vector<double>& solution, 
        Function<dim>& f,
        unsigned int component);

    /**
     * Computes the global integral of the solution vector for the given species.
     */
    Tensor<1, 5, double> compute_global_integral(
        LinearAlgebra::distributed::Vector<double>& solution,
        unsigned int species_index);

   private:
    std::shared_ptr<NodalDGDiscretization<dim>> discretization;
};

}  // namespace five_moment
}  // namespace warpii
