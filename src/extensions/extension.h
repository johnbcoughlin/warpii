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

/**
 * Abstract class for extensions that construct their own mesh using
 * deal.II's Triangulation manipulation functions.
 */
template <int dim>
class GridExtension {
    public:
    virtual ~GridExtension() = default;

    /**
     * Declare any parameters required for the triangulation.
     *
     * @param prm: Will be scoped to the `geometry` subsection.
     */
    virtual void declare_geometry_parameters(dealii::ParameterHandler& prm);

    /**
     * Populate the given triangulation with vertex and cell information.
     *
     * @param tria: The output triangulation
     * @param prm: Will be scoped to the `geometry` subsection. Can be used to retrieve
     * parameters declared in declare_geometry_parameters().
     */
    virtual void populate_triangulation(dealii::Triangulation<dim>& tria,
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

template <typename ExtType>
std::shared_ptr<ExtType> unwrap_extension(std::shared_ptr<warpii::Extension> ext) {
    if (!ext) {
        return std::make_shared<ExtType>();
    }
    if (auto result = std::dynamic_pointer_cast<ExtType>(ext)) {
        return result;
    }
    return std::make_shared<ExtType>();
}

}  // namespace warpii
