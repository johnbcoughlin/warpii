#include "nodal_dg_discretization.h"

namespace warpii {

template <int dim>
void NodalDGDiscretization<dim>::reinit() {
    dof_handler.distribute_dofs(fe);

    const std::vector<const DoFHandler<dim> *> dof_handlers = {&dof_handler};
    const AffineConstraints<double> dummy;
    const std::vector<const AffineConstraints<double> *> constraints = {&dummy};
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

template <int dim>
void NodalDGDiscretization<dim>::perform_allocation(
    LinearAlgebra::distributed::Vector<double> &solution) {
    mf.initialize_dof_vector(solution);
}

template class NodalDGDiscretization<1>;
template class NodalDGDiscretization<2>;

}
