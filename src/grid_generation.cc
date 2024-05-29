#include "grid_generation.h"
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/base/point.h>
#include "utilities.h"

using namespace dealii;
namespace warpii {

void rectangular_hyper_shell(Triangulation<2>& tria,
        const Point<2> left,
        const Point<2> right,
        const double x_width,
        const double y_width,
        const std::vector<unsigned int>& n_cells) {

    unsigned int nx = n_cells.at(0);
    unsigned int ny = n_cells.at(1);

    double Lx = (right - left)[0];
    double Ly = (right - left)[1];

    double dx_out = Lx / nx;
    double dy_out = Ly / ny;
    double dx_in = (Lx - 2*x_width) / nx;
    double dy_in = (Ly - 2*y_width) / ny;

    // Number of cells
    unsigned int N = 2*nx + 2*ny;

    std::vector<Point<2>> vertices(2*N);
    // Bottom edge
    for (unsigned int i = 0; i < nx; i++) {
        vertices[i] = left + Point<2>(i * dx_out, 0.0);
        vertices[i + N] = left + Point<2>(x_width + i * dx_in, y_width);
    }
    // Right edge
    for (unsigned int i = 0; i < ny; i++) {
        vertices[nx+i] = left + Point<2>(Lx, i*dy_out);
        vertices[nx + i + N] = left + Point<2>(Lx - x_width, y_width + i * dy_in);
    }
    // Top edge
    for (unsigned int i = 0; i < nx; i++) {
        vertices[nx + ny + i] = right + Point<2>(-(i * dx_out), 0.0);
        vertices[nx + ny + i + N] = right + Point<2>(-x_width - i * dx_in, -y_width);
    }
    // Left edge
    for (unsigned int i = 0; i < ny; i++) {
        vertices[2*nx + ny + i] = left + Point<2>(0.0, Ly - i * dy_out);
        vertices[2*nx + ny + i + N] = left + Point<2>(x_width, Ly - y_width - i * dy_in);
    }

    std::vector<CellData<2>> cells(N, CellData<2>());

    for (unsigned int i = 0; i < N; i++) {
        cells[i].vertices[0] = i;
        cells[i].vertices[1] = (i+1) % N;
        cells[i].vertices[2] = N + i;
        cells[i].vertices[3] = N + ((i + 1) % N);

        cells[i].material_id = 0;
    }

    tria.create_triangulation(vertices, cells, SubCellData());

    tria.set_all_manifold_ids(0);
}

void concentric_rectangular_hyper_shells(Triangulation<2>& tria,
        const Point<2> left,
        const Point<2> right,
        const std::vector<double> &x_widths,
        const std::vector<double> &y_widths,
        const std::vector<unsigned int>& n_cells) {
    AssertThrow(x_widths.size() == y_widths.size(), ExcMessage("x_widths and y_widths must have the same length"));

    Point<2> shell_left = left;
    Point<2> shell_right = right;

    unsigned int n_shells = x_widths.size();
    for (unsigned int i = 0; i < n_shells; i++) {
        double x_width = x_widths[i];
        double y_width = y_widths[i];
        SHOW(x_width);
        SHOW(y_width);
        Triangulation<2> shell_tria;
        std::cout << shell_left << std::endl;
        std::cout << shell_right << std::endl;
        rectangular_hyper_shell(shell_tria, shell_left, shell_right,
                x_width, y_width, n_cells);
        shell_left += Point<2>(x_width, y_width);
        shell_right -= Point<2>(x_width, y_width);

        double grid_vertex_tolerance = 0.1 * std::min(x_widths[i], y_widths[i]);
        //Triangulation<2> temp;
        //temp.copy_triangulation(tria);
        //tria.clear();
        GridGenerator::merge_triangulations(
                shell_tria, tria, tria, grid_vertex_tolerance);
    }
}

}
