#include <deal.II/base/logstream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/grid/grid_generator.h>

#include <iostream>
#include <src/five_moment/extension.h>
#include <src/grid_generation.h>
#include <src/warpii.h>

using namespace dealii;
using namespace warpii;

class ForwardFacingStep : public warpii::five_moment::Extension<2> {
    void declare_geometry_parameters(dealii::ParameterHandler& prm) override {
        prm.declare_entry("RefinementFactor", "1", Patterns::Integer(1));
    }

    void populate_triangulation(
            dealii::Triangulation<2>& tria,
            const ParameterHandler& prm
            ) override {

    unsigned int refinement_factor = prm.get_integer("RefinementFactor");

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

    tria.clear();
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
            rectangular_tria, cells_to_remove, tria);

    double tol = std::sqrt(std::numeric_limits<double>::epsilon());
    for (const auto &cell : tria.active_cell_iterators()) {
        for (const unsigned int face_n : GeometryInfo<2>::face_indices()) {
            const auto face = cell->face(face_n);
            if (!face->at_boundary()) {
                continue;
            }
            if (std::abs(face->center()[0]) < tol) {
                // Left
                face->set_boundary_id(0);
            } else if (std::abs(face->center()[1]) < tol) {
            // Bottom
                face->set_boundary_id(1);
            } else if (std::abs(face->center()[0] - 3*dx) < tol && face->center()[1] < dy) {
            // Corner left leg
                face->set_boundary_id(2);
            } else if (std::abs(face->center()[1] - dy) < tol && face->center()[0] > 3*dx) {
            // Corner long leg
                face->set_boundary_id(3);
            } else if (std::abs(face->center()[0] - Lx) < tol) {
                // Right
                face->set_boundary_id(4);
            } else if (std::abs(face->center()[1] - Ly) < tol) {
                // Top
                face->set_boundary_id(5);
            }
        }
    }
    }
};

int main(int argc, char **argv) {
    using namespace dealii;

    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    deallog.depth_console(0);

    Warpii warpii_obj = Warpii::create_from_cli(argc, argv, std::make_shared<ForwardFacingStep>());

    warpii_obj.run();

    return 0;
}

