#pragma once

#include <deal.II/base/parameter_handler.h>
#include <deal.II/grid/tria.h>

using namespace dealii;

namespace warpii {

/**
 * Abstract superclass of all extension mechanisms for WarpII.
 *
 * Specific apps provide subclasses of this class.
 */
class Extension {
   public:
    virtual ~Extension() = default;
};

template <int dim>
class GridExtension {
    public:
    virtual ~GridExtension() = default;

    /**
     * Declare any parameters required for the triangulation.
     */
    virtual void declare_geometry_parameters(dealii::ParameterHandler& prm);

    virtual void populate_triangulation(dealii::Triangulation<dim>&,
                                        const dealii::ParameterHandler& prm);
};

template <int dim>
void GridExtension<dim>::declare_geometry_parameters(ParameterHandler &) {
}

template <int dim>
void GridExtension<dim>::populate_triangulation(Triangulation<dim>&, const ParameterHandler &) {
}

template class GridExtension<1>;
template class GridExtension<2>;

}  // namespace warpii
