#pragma once

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q.h>

#include "../grid.h"

using namespace dealii;

namespace warpii {
namespace five_moment {

template <int dim>
class FiveMomentDGSolution {
   public:
    FiveMomentDGSolution(std::shared_ptr<Grid<dim>> grid, 
            unsigned int n_components,
            unsigned int fe_degree)
        : grid(std::move(grid)), mapping(fe_degree), fe(FE_DGQ<dim>(fe_degree) ^ (n_components)) {}

   private:
    std::shared_ptr<Grid<dim>> grid;
    MappingQ<dim> mapping;
    FESystem<dim> fe;
    DoFHandler<dim> dof_handler;

    LinearAlgebra::distributed::Vector<double> solution;
};

}  // namespace five_moment
}  // namespace warpii
