#pragma once

#include <deal.II/grid/tria.h>
#include <deal.II/base/point.h>

using namespace dealii;

namespace warpii {

void rectangular_hyper_shell(Triangulation<2>& tria,
        const Point<2> left,
        const Point<2> right,
        const double x_width,
        const double y_width,
        const std::vector<unsigned int>& n_cells);

void concentric_rectangular_hyper_shells(Triangulation<2>& tria,
        const Point<2> left,
        const Point<2> right,
        const std::vector<double> &x_widths,
        const std::vector<double> &y_widths,
        const std::vector<unsigned int>& n_cells);

}
