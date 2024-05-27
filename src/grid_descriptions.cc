#include "grid_descriptions.h"
#include "grid_generation.h"

#include <deal.II/base/patterns.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

using namespace dealii;

namespace warpii {

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
void HyperRectangleDescription<dim>::declare_parameters(ParameterHandler& prm) {
    declare_left_right<dim>(prm);
    declare_nx<dim>(prm);
    prm.declare_entry("periodic_dimensions", "x,y,z",
                      Patterns::MultipleSelection("x|y|z"));
}

template <int dim>
std::unique_ptr<HyperRectangleDescription<dim>>
HyperRectangleDescription<dim>::create_from_parameters(ParameterHandler& prm) {
    std::array<unsigned int, dim> nx =
        Patterns::Tools::Convert<std::array<unsigned int, dim>>::to_value(
            prm.get("nx"));
    Point<dim> left =
        Patterns::Tools::Convert<Point<dim>>::to_value(prm.get("left"));
    Point<dim> right =
        Patterns::Tools::Convert<Point<dim>>::to_value(prm.get("right"));
    std::string periodic_dims = prm.get("periodic_dimensions");

    return std::make_unique<HyperRectangleDescription<dim>>(nx, left, right,
                                                            periodic_dims);
}

template <int dim>
void HyperRectangleDescription<dim>::reinit(Triangulation<dim>& triangulation) {
    std::vector<unsigned int> subdivisions =
        std::vector(nx.data(), nx.data() + dim);
    GridGenerator::subdivided_hyper_rectangle(triangulation, subdivisions, left,
                                              right, true);

    std::vector<GridTools::PeriodicFacePair<
        typename parallel::distributed::Triangulation<dim>::cell_iterator>>
        matched_pairs;
    if (periodic_dims.find("x") != std::string::npos) {
        GridTools::collect_periodic_faces(triangulation, 0, 1, 0,
                                          matched_pairs);
    }
    if (dim >= 2 && periodic_dims.find("y") != std::string::npos) {
        GridTools::collect_periodic_faces(triangulation, 2, 3, 1,
                                          matched_pairs);
    }
    if (dim >= 3 && periodic_dims.find("z") != std::string::npos) {
        GridTools::collect_periodic_faces(triangulation, 4, 5, 2,
                                          matched_pairs);
    }
    triangulation.add_periodicity(matched_pairs);
}

template class HyperRectangleDescription<1>;
template class HyperRectangleDescription<2>;

template <int dim>
void ForwardFacingStepDescription<dim>::declare_parameters(
    ParameterHandler& prm) {
    prm.declare_entry("RefinementFactor", "1", Patterns::Integer(0));
}

template <int dim>
std::unique_ptr<ForwardFacingStepDescription<dim>>
ForwardFacingStepDescription<dim>::create_from_parameters(
    ParameterHandler& prm) {
    return std::make_unique<ForwardFacingStepDescription<dim>>(
        prm.get_integer("RefinementFactor"));
}

template <int dim>
void ForwardFacingStepDescription<dim>::reinit(Triangulation<dim>& ) {
    Assert(false, ExcMessage("ForwardFacingStep only supported in 2D"));
}

template <>
void ForwardFacingStepDescription<2>::reinit(
    Triangulation<2>& triangulation) {

    double Lx = 3.0;
    double Ly = 1.0;

    unsigned int nx = 15;
    unsigned int ny = 5;

    double dx = Lx / nx;
    double dy = Ly / ny;

    std::vector<std::vector<double>> step_sizes;
    step_sizes.push_back({dx/2.0, dx/4.0, dx/4.0, dx/4.0, dx/4.0, dx/2.0});
    step_sizes.push_back({dy/2.0, dy/4.0, dy/4.0, dy/4.0, dy/4.0, dy/2.0});

    Triangulation<2> corner_tria;
    
    std::vector<unsigned int> n_cells = {2*refinement_factor, 2*refinement_factor};
    std::vector<double> x_widths;
    std::vector<double> y_widths;

    double c = 0.0;
    for (unsigned int i = 0; i < refinement_factor; i++) {
        c += 1.0 / std::sqrt(i + 1.0);
    }
    for (unsigned int i = 0; i < refinement_factor; i++) {
        x_widths.push_back(0.75 * dx * (1.0 / std::sqrt(i + 1.0)) / c);
        y_widths.push_back(0.75 * dy * (1.0 / std::sqrt(i + 1.0)) / c);
    }

    concentric_rectangular_hyper_shells(corner_tria,
            Point<2>(2*dx, 0.0), Point<2>(4*dx, 2*dy),
            x_widths,
            y_widths,
            n_cells);

    Triangulation<2> corner_square_tria;
    std::vector<unsigned int> n_corner_cells = {2*refinement_factor, 2*refinement_factor};
    GridGenerator::subdivided_hyper_rectangle(corner_square_tria,
            n_corner_cells, Point<2>(2.75*dx, .75*dy), Point<2>(3.25*dx, 1.25*dy));
    GridGenerator::merge_triangulations(corner_square_tria, corner_tria, corner_tria);

    std::vector<unsigned int> bulk_subdivisions = {nx*refinement_factor, ny*refinement_factor};

    triangulation.clear();
    Triangulation<2> bulk_tria;
    GridGenerator::subdivided_hyper_rectangle(
            bulk_tria, bulk_subdivisions, Point<2>(0.0, 0.0), Point<2>(Lx, Ly));

    std::set<Triangulation<2>::active_cell_iterator> cells_to_remove;
    for (const auto &cell : bulk_tria.active_cell_iterators()) {
        Point<2> cell_center = cell->center();
        if (cell_center[0] > 2 * dx && cell_center[0] < 4*dx && cell_center[1] < 2*dy) {
            cells_to_remove.insert(cell);
        }
    }
    Triangulation<2> rectangular_tria;
    GridGenerator::create_triangulation_with_removed_cells(
            bulk_tria, cells_to_remove, rectangular_tria);
    GridGenerator::merge_triangulations(
            rectangular_tria, corner_tria, rectangular_tria);

    cells_to_remove.clear();
    for (const auto& cell : rectangular_tria.active_cell_iterators()) {
        if (cell->center()[0] > 3*dx && cell->center()[1] < dy) {
            cells_to_remove.insert(cell);
        }
    }
    GridGenerator::create_triangulation_with_removed_cells(
            rectangular_tria, cells_to_remove, triangulation);
}

template class ForwardFacingStepDescription<1>;
template class ForwardFacingStepDescription<2>;

}  // namespace warpii
