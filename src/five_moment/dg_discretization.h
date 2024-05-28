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
#include "../grid.h"

namespace warpii {
namespace five_moment {
template <int dim>
class FiveMomentDGDiscretization {
   public:
    FiveMomentDGDiscretization(std::shared_ptr<Grid<dim>> grid,
                               unsigned int n_components,
                               unsigned int fe_degree)
        : fe_degree(fe_degree),
        n_components(n_components),
          grid(grid),
          mapping(fe_degree),
          fe(FE_DGQ<dim>(fe_degree) ^ (n_components)),
          dof_handler(grid->triangulation),
          fe_values(mapping, fe, QGaussLobatto<dim>(fe_degree+1), UpdateFlags::update_values),
          dummy_fe_collection(fe.base_element(0)),
          dummy_q_collection(fe_values.get_quadrature())
    {}

    void reinit();

    void perform_allocation(
        LinearAlgebra::distributed::Vector<double> &solution);

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
    Tensor<1, dim+2, double> compute_global_integral(
        LinearAlgebra::distributed::Vector<double>& solution,
        unsigned int species_index);

    unsigned int get_fe_degree() {
        return fe_degree;
    }
    DoFHandler<dim>& get_dof_handler() {
        return dof_handler;
    }
    Mapping<dim>& get_mapping() {
        return mapping;
    }
    FESystem<dim>& get_fe() {
        return fe;
    }
    MatrixFree<dim>& get_matrix_free() {
        return mf;
    }
    hp::FECollection<dim>& get_dummy_fe_collection() {
        return dummy_fe_collection;
    }
    hp::QCollection<dim>& get_dummy_q_collection() {
        return dummy_q_collection;
    }

    void build_data_out_patches(DataOut<dim>& data_out) {
        data_out.build_patches(mapping, fe.degree, DataOut<dim>::curved_inner_cells);
    }

   private:
    unsigned int fe_degree;
    unsigned int n_components;
    std::shared_ptr<Grid<dim>> grid;
    MappingQ<dim> mapping;
    FESystem<dim> fe;
    DoFHandler<dim> dof_handler;
    FEValues<dim> fe_values;
    hp::FECollection<dim> dummy_fe_collection;
    hp::QCollection<dim> dummy_q_collection;
   public:
    MatrixFree<dim> mf;
};

}  // namespace five_moment
}  // namespace warpii
