#pragma once

#include <deal.II/base/parameter_handler.h>
#include <deal.II/fe/mapping_q.h>

#include <memory>

#include "../grid.h"
#include "fluid_flux_dg_operator.h"
#include "species.h"

using namespace dealii;

namespace warpii {
namespace five_moment {

template <int dim>
class FiveMomentDGOperator {
   public:
    FiveMomentDGOperator(std::vector<Species<dim>> species) : species(species) {}

    void reinit(unsigned int fe_degree,
                MappingQ<dim> mapping,
                DoFHandler<dim> dof_handler);

   private:
    std::vector<Species<dim>> species;
    MatrixFree<dim, double> mf;
    FluidFluxDGOperator<dim> fluid_flux_operator;
    // FluidSourceDGOperator<dim> fluid_source_operator;
    // FieldFluxDGOperator<dim> field_flux_operator;
    // FieldSourceDGOperator<dim> field_source_operator;
};

template <int dim>
void FiveMomentDGOperator<dim>::reinit(
    unsigned int fe_degree, 
    MappingQ<dim> mapping,
    DoFHandler<dim> dof_handler) {

    auto dof_handlers = { dof_handler };
    const AffineConstraints<double> dummy;
    const std::vector<const AffineConstraints<double>*> constraints = {&dummy};
    const std::vector<Quadrature<1>> quadratures = {
        QGauss<1>(fe_degree + 2), QGaussLobatto<1>(fe_degree + 1)};

    typename MatrixFree<dim, double>::AdditionalData additional_data;
    additional_data.mapping_update_flags =
        (update_gradients | update_JxW_values | update_quadrature_points |
         update_values);
    additional_data.mapping_update_flags_inner_faces =
        (update_JxW_values | update_quadrature_points | update_normal_vectors |
         update_values);
    additional_data.mapping_update_flags_boundary_faces =
        (update_JxW_values | update_quadrature_points | update_normal_vectors |
         update_values);
    additional_data.tasks_parallel_scheme =
        MatrixFree<dim, double>::AdditionalData::none;

    mf.reinit(mapping, dof_handlers, constraints, quadratures, additional_data);
}

}  // namespace five_moment
}  // namespace warpii
