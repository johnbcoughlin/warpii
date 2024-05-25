#pragma once

#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/patterns.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_tools.h>
#include <memory>

namespace warpii {
using namespace dealii;

template <int dim>
class Grid {
   public:
    Grid(Point<dim> left, Point<dim> right, std::array<unsigned int, dim> nx,
            std::string periodic_dims)
        : left(left), right(right), nx(nx), periodic_dims(periodic_dims) {}

    static void declare_parameters(ParameterHandler& prm);

    static std::shared_ptr<Grid<dim>> create_from_parameters(
        ParameterHandler& prm);

    void reinit();

    Triangulation<dim> triangulation;

   private:
    Point<dim> left;
    Point<dim> right;
    std::array<unsigned int, dim> nx;
    std::string periodic_dims;
};

template <int dim>
Point<dim> default_right_point() {
    if (dim == 1) {
        return Point<dim>(1.0);
    } else if (dim == 2) {
        return Point<dim>(1.0, 1.0);
    } else {
        return Point<dim>(1.0, 1.0, 1.0);
    }
}

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
        Point<dim> pt1 = default_right_point<dim>();
        prm.declare_entry("right", PointPattern::to_string(pt1), *PointPattern::to_pattern());

        prm.declare_entry("periodic_dimensions", "x,y,z", Patterns::MultipleSelection("x|y|z"));
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
    std::string periodic_dims = prm.get("periodic_dimensions");

    auto result = std::make_shared<Grid<dim>>(left, right, nx, periodic_dims);
    prm.leave_subsection();  // geometry
    return result;
}

template <int dim>
void Grid<dim>::reinit() {
    std::vector<unsigned int> subdivisions = std::vector(nx.data(), nx.data()+dim);
    GridGenerator::subdivided_hyper_rectangle(triangulation, subdivisions, left, right, true);

    std::vector<GridTools::PeriodicFacePair<
        typename parallel::distributed::Triangulation<dim>::cell_iterator>>
        matched_pairs;
    if (periodic_dims.find("x") != std::string::npos) {
        GridTools::collect_periodic_faces(triangulation, 0, 1, 0, matched_pairs);
    }
    if (dim >= 2 && periodic_dims.find("y") != std::string::npos) {
        GridTools::collect_periodic_faces(triangulation, 2, 3, 1, matched_pairs);
    }
    if (dim >= 3 && periodic_dims.find("z") != std::string::npos) {
        GridTools::collect_periodic_faces(triangulation, 4, 5, 2, matched_pairs);
    }
    triangulation.add_periodicity(matched_pairs);
}

}  // namespace warpii
