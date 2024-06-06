#pragma once
#include <deal.II/base/parameter_handler.h>
#include <deal.II/grid/tria.h>
#include "extensions/extension.h"

using namespace dealii;
namespace warpii {

template <int dim>
Point<dim> default_right_point();

/**
 * Declare a pair of Point<dim> parameters named "left" and "right"
 */
template <int dim>
void declare_left_right(ParameterHandler& prm) {
    using PointPattern = Patterns::Tools::Convert<Point<dim>>;
    Point<dim> pt = Point<dim>();
    prm.declare_entry("left", PointPattern::to_string(pt),
                      *PointPattern::to_pattern());
    Point<dim> pt1 = default_right_point<dim>();
    prm.declare_entry("right", PointPattern::to_string(pt1),
                      *PointPattern::to_pattern());
}

/**
 * Declare a std::array<unsigned int, dim> parameter named "nx"
 */
template <int dim>
void declare_nx(ParameterHandler& prm) {
    using ArrayPattern =
        Patterns::Tools::Convert<std::array<unsigned int, dim>>;
    std::array<unsigned int, dim> default_nx;
    default_nx.fill(1);
    prm.declare_entry("nx", ArrayPattern::to_string(default_nx),
                      *ArrayPattern::to_pattern());
}

template <int dim>
void declare_int_array(ParameterHandler &prm, std::string name) {
    using ArrayPattern =
        Patterns::Tools::Convert<std::array<int, dim>>;
    std::array<int, dim> default_val;
    default_val.fill(0);
    prm.declare_entry(name, ArrayPattern::to_string(default_val),
                      *ArrayPattern::to_pattern());
}

/**
 * A description of a grid to be generated.
 *
 * Abstract class.
 *
 * Many of the subclasses of GridDescription contain parameters required
 * to form calls to one of the GridGenerators provided by deal.II.
 * See [the GridGenerators
 * documentation](https://www.dealii.org/developer/doxygen/deal.II/namespaceGridGenerator.html)
 * for examples.
 */
template <int dim>
class GridDescription {
   public:
    virtual ~GridDescription() = default;

    /**
     * Reinitialize the given triangulation with the current description.
     * This likely involves making a call to a
     * [GridGenerator](https://www.dealii.org/developer/doxygen/deal.II/namespaceGridGenerator.html)
     * with the stored parameters.
     */
    virtual void reinit(Triangulation<dim>& tri);
};

template <int dim>
class ExtensionGridDescription : public GridDescription<dim> {
    public:
        ExtensionGridDescription(ParameterHandler& prm,
                std::shared_ptr<GridExtension<dim>> ext):
            prm(prm), ext(ext)

    {}

        static void declare_parameters(ParameterHandler& prm);

        static std::unique_ptr<ExtensionGridDescription<dim>> create_from_parameters(
                ParameterHandler& prm, std::shared_ptr<GridExtension<dim>> ext) {
            return std::make_unique<ExtensionGridDescription<dim>>(prm, ext);
        }

        void reinit(Triangulation<dim>& tri) override;

    private:
        ParameterHandler& prm;
        std::shared_ptr<GridExtension<dim>> ext;
};

template <int dim>
void ExtensionGridDescription<dim>::reinit(Triangulation<dim>& tria) {
    prm.enter_subsection("geometry");
    ext->populate_triangulation(tria, prm);
    prm.leave_subsection();
}

/**
 * Description of a deal.II
 * [subdivided_hyper_rectangle](https://www.dealii.org/developer/doxygen/deal.II/namespaceGridGenerator.html#ac76417d7404b75cf53c732f456e6e971)
 * call
 */
template <int dim>
class HyperRectangleDescription : public GridDescription<dim> {
   public:
    HyperRectangleDescription(std::array<unsigned int, dim> nx, Point<dim> left,
                              Point<dim> right, std::string periodic_dims)
        : nx(nx), left(left), right(right), periodic_dims(periodic_dims) {}

    static void declare_parameters(ParameterHandler& prm);

    static std::unique_ptr<HyperRectangleDescription<dim>>
    create_from_parameters(ParameterHandler& prm);

    void reinit(Triangulation<dim>& tri) override;

   private:
    std::array<unsigned int, dim> nx;
    Point<dim> left;
    Point<dim> right;
    std::string periodic_dims;
};

}  // namespace warpii
