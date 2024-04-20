#pragma once

#include <deal.II/base/parameter_handler.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <memory>

namespace warpii {
using namespace dealii;

template <int dim>
class Grid {
   public:
    Grid(Point<dim> left, Point<dim> right, std::array<unsigned int, dim> nx)
        : left(left), right(right), nx(nx) {}

    static void declare_parameters(ParameterHandler& prm);

    static std::shared_ptr<Grid<dim>> create_from_parameters(
        ParameterHandler& prm);

    void reinit();

   private:
    Point<dim> left;
    Point<dim> right;
    std::array<unsigned int, dim> nx;
    Triangulation<dim> triangulation;
};

template <int dim>
void Grid<dim>::declare_parameters(ParameterHandler& prm) {
    using ArrayPattern = Patterns::Tools::Convert<
                              std::array<unsigned int, dim>>;
    using PointPattern = Patterns::Tools::Convert<Point<dim>>;
    prm.enter_subsection("geometry");
    {
        std::array<unsigned int, dim> default_nx;
        default_nx.fill(1);
        prm.declare_entry("nx", ArrayPattern::to_string(default_nx),
                          *ArrayPattern::to_pattern());
        Point<dim> pt = Point<dim>();
        prm.declare_entry("left", PointPattern::to_string(pt), *PointPattern::to_pattern());
        prm.declare_entry("right", PointPattern::to_string(pt), *PointPattern::to_pattern());
    }
    prm.leave_subsection();  // geometry
}

template <int dim>
std::shared_ptr<Grid<dim>> Grid<dim>::create_from_parameters(
    ParameterHandler& prm) {
    prm.enter_subsection("geometry");
    Triangulation<dim> tri;
    std::array<unsigned int, dim> nx = Patterns::Tools::Convert<std::array<unsigned int, dim>>::to_value(prm.get("nx"));
    Point<dim> left = Patterns::Tools::Convert<Point<dim>>::to_value(prm.get("left"));
    Point<dim> right = Patterns::Tools::Convert<Point<dim>>::to_value(prm.get("right"));

    auto result = std::make_shared<Grid<dim>>(left, right, nx);
    prm.leave_subsection();  // geometry
    return result;
}

template <int dim>
void Grid<dim>::reinit() {
    GridGenerator::subdivided_hyper_rectangle(triangulation, left, right, nx);
}

}  // namespace warpii
