#pragma once
#include <deal.II/base/parameter_handler.h>
#include <deal.II/grid/tria.h>

using namespace dealii;
namespace warpii {

template <int dim> Point<dim> default_right_point();

/**
 * A description of a grid to be generated.
 *
 * Abstract class.
 *
 * Many of the subclasses of GridDescription contain parameters required
 * to form calls to one of the GridGenerators provided by deal.II.
 * See [the GridGenerators documentation](https://www.dealii.org/developer/doxygen/deal.II/namespaceGridGenerator.html)
 * for examples.
 */
template <int dim>
class GridDescription {
    public:
        virtual ~GridDescription() = default;

        /**
         * Reinitialize the given triangulation with the current description.
         * This likely involves making a call to a [GridGenerator](https://www.dealii.org/developer/doxygen/deal.II/namespaceGridGenerator.html)
         * with the stored parameters.
         */
        virtual void reinit(Triangulation<dim>& tri);
};

/**
 * Description of a deal.II [subdivided_hyper_rectangle](https://www.dealii.org/developer/doxygen/deal.II/namespaceGridGenerator.html#ac76417d7404b75cf53c732f456e6e971) call
 */
template <int dim>
class HyperRectangleDescription : public GridDescription<dim> {
   public:
    HyperRectangleDescription(
    std::array<unsigned int, dim> nx,
    Point<dim> left,
    Point<dim> right,
    std::string periodic_dims):
        nx(nx), left(left), right(right),
        periodic_dims(periodic_dims)
    {}

    static void declare_parameters(ParameterHandler& prm);

    static std::unique_ptr<HyperRectangleDescription<dim>> create_from_parameters(
        ParameterHandler& prm);

    void reinit(Triangulation<dim>& tri) override;

   private:
    std::array<unsigned int, dim> nx;
    Point<dim> left;
    Point<dim> right;
    std::string periodic_dims;
};

template <int dim>
class HyperLDescription {
   public:
    static void declare_parameters(ParameterHandler& prm);

    static HyperLDescription<dim> create_from_parameters(ParameterHandler& prm);
};

}  // namespace warpii
