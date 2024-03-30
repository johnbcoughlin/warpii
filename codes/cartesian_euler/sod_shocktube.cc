#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/time_stepping.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/vectorization.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/numerics/data_out.h>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>

#include "CartesianEulerOperator.h"
#include "CartesianEulerPostProcessor.h"
#include "rk.h"

namespace CartesianEuler {
using namespace dealii;

constexpr unsigned int fe_degree = 4;
// constexpr LowStorageRungeKuttaScheme lsrk_scheme = stage_5_order_4;
const double courant_number = 1.0 / 6.0 / 2.0;
constexpr double gamma = 5.0 / 3.0;
constexpr double final_time = 0.5;
constexpr double output_tick = 0.001;

class SodShocktubeInitialCondition : public Function<1> {
   public:
    SodShocktubeInitialCondition(const double gamma, const double ratio,
                                 const double time)
        : Function<1>(3, time), gamma(gamma), ratio(ratio) {}

    virtual double value(const Point<1> &p,
                         const unsigned int component = 0) const override;

   private:
    double gamma;
    double ratio;
};

double SodShocktubeInitialCondition::value(const Point<1> &p,
                                           const unsigned int component) const {
    double x = p[0];

    if (component == 0) {
        return x < 0.0 ? 1.0 : 0.125 / ratio;
    } else if (component == 1) {
        return 0.0;
    } else if (component == 2) {
        double pressure = x < 0.0 ? 1.0 : 0.1 / ratio;
        double energy = pressure / (gamma - 1.0);
        return energy;
    } else {
        Assert(false, ExcNotImplemented());
        return 0.0;
    }
}

class SodShocktubeProblem {
   public:
    SodShocktubeProblem(double gamma, double ratio);

    void run();

   private:
    double gamma;
    double ratio;
    double time;
    double time_step;

    ConditionalOStream pcout;
    Triangulation<1> triangulation;
    FESystem<1> fe;
    MappingQ<1> mapping;
    DoFHandler<1> dof_handler;

    TimerOutput timer;

    CartesianEulerOperator<1, fe_degree, fe_degree + 1> euler_operator;

    void make_grid_and_dofs();

    void output_results(const unsigned int result_number);

    LinearAlgebra::distributed::Vector<Number> solution;

    void project(const Function<1> &function,
                 LinearAlgebra::distributed::Vector<Number> &solution) const;
};

SodShocktubeProblem::SodShocktubeProblem(double gamma, double ratio)
    : gamma(gamma),
      ratio(ratio),
      time(0.0),
      time_step(0.0),
      pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
      fe(FE_DGQ<1>(fe_degree) ^ (3)),
      mapping(fe_degree),
      dof_handler(triangulation),
      timer(pcout, TimerOutput::never, TimerOutput::wall_times),
      euler_operator(timer, gamma) {}

void SodShocktubeProblem::make_grid_and_dofs() {
    Point<1> left(-0.5);
    Point<1> right(0.5);

    GridGenerator::subdivided_hyper_rectangle(triangulation, {16}, left, right,
                                              true);
    triangulation.refine_global(2);
    const auto ic = SodShocktubeInitialCondition(gamma, ratio, 0.0);

    euler_operator.bc_map().set_supersonic_outflow_boundary(0);
    euler_operator.bc_map().set_supersonic_outflow_boundary(1);

    dof_handler.distribute_dofs(fe);

    euler_operator.reinit(mapping, dof_handler);
    euler_operator.initialize_vector(solution);
}

void SodShocktubeProblem::output_results(const unsigned int result_number) {
    TimerOutput::Scope t(timer, "output");

    CartesianEulerPostprocessor<1> postprocessor(gamma);
    DataOut<1> data_out;

    DataOutBase::VtkFlags flags;
    flags.write_higher_order_cells = false;
    data_out.set_flags(flags);

    data_out.attach_dof_handler(dof_handler);
    {
        std::vector<std::string> names;
        names.emplace_back("density");
        for (unsigned int d = 0; d < 1; ++d) names.emplace_back("momentum");
        names.emplace_back("energy");

        std::vector<DataComponentInterpretation::DataComponentInterpretation>
            interpretation;

        interpretation.push_back(
            DataComponentInterpretation::component_is_scalar);
        for (unsigned int d = 0; d < 1; ++d)
            interpretation.push_back(
                DataComponentInterpretation::component_is_part_of_vector);
        interpretation.push_back(
            DataComponentInterpretation::component_is_scalar);

        data_out.add_data_vector(dof_handler, solution, names, interpretation);
    }
    data_out.add_data_vector(solution, postprocessor);

    Vector<double> mpi_owner(triangulation.n_active_cells());
    mpi_owner = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    data_out.add_data_vector(mpi_owner, "owner");

    data_out.build_patches(mapping, fe.degree, DataOut<1>::curved_inner_cells);

    const std::string filename =
        "solution_" + Utilities::int_to_string(result_number, 3) + ".vtu";
    data_out.write_vtu_in_parallel(filename, MPI_COMM_WORLD);
}

void SodShocktubeProblem::run() {
    {
        const unsigned int n_vect_number = VectorizedArray<Number>::size();
        const unsigned int n_vect_bits = 8 * sizeof(Number) * n_vect_number;

        pcout << "Running with "
              << Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)
              << " MPI processes" << std::endl;
        pcout << "Vectorization over " << n_vect_number << ' '
              << (std::is_same_v<Number, double> ? "doubles" : "floats")
              << " = " << n_vect_bits << " bits ("
              << Utilities::System::get_current_vectorization_level() << ')'
              << std::endl;
    }

    make_grid_and_dofs();

    // const LowStorageRungeKuttaIntegrator integrator(lsrk_scheme);
    SSPRK2Integrator<Number,
                     CartesianEulerOperator<1, fe_degree, fe_degree + 1>>
        integrator;
    integrator.reinit(solution, 3);

    LinearAlgebra::distributed::Vector<Number> rk_register_1;
    LinearAlgebra::distributed::Vector<Number> rk_register_2;
    rk_register_1.reinit(solution);
    rk_register_2.reinit(solution);

    euler_operator.project(SodShocktubeInitialCondition(gamma, ratio, 0.0),
                           solution);

    double min_vertex_distance = std::numeric_limits<double>::max();
    for (const auto &cell : triangulation.active_cell_iterators())
        if (cell->is_locally_owned())
            min_vertex_distance =
                std::min(min_vertex_distance, cell->minimum_vertex_distance());
    min_vertex_distance =
        Utilities::MPI::min(min_vertex_distance, MPI_COMM_WORLD);

    double max_time_step = 1e-3;
    time_step = courant_number /
                euler_operator.compute_cell_transport_speed(solution);
    time_step = std::min(time_step, max_time_step);
    pcout << "Time step size: " << time_step
          << ", minimal h: " << min_vertex_distance
          << ", initial transport scaling: "
          << 1. / euler_operator.compute_cell_transport_speed(solution)
          << std::endl
          << std::endl;

    output_results(0);

    while (time < final_time - 1e-12) {
        time_step = Utilities::truncate_to_n_digits(courant_number /
                euler_operator.compute_cell_transport_speed(solution), 3);

        time_step = std::min(time_step, max_time_step);

        {
            TimerOutput::Scope t(timer, "rk time stepping total");
            integrator.evolve_one_time_step(euler_operator, solution, time_step,
                                            time);
            // Assert(false, ExcInternalError());
        }

        time += time_step;

        if (static_cast<int>(time / output_tick) !=
                static_cast<int>((time - time_step) / output_tick) ||
            time >= final_time - 1e-12) {
            output_results(
                static_cast<unsigned int>(std::round(time / output_tick)));
            pcout << "t = " << time << std::endl;
        }
    }

    timer.print_wall_time_statistics(MPI_COMM_WORLD);
    pcout << std::endl;
}

}  // namespace CartesianEuler

int main(int argc, char **argv) {
    using namespace CartesianEuler;
    using namespace dealii;

    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    try {
        deallog.depth_console(0);

        double ratio = std::atof(argv[1]);

        SodShocktubeProblem sod_problem(gamma, ratio);
        sod_problem.run();
    } catch (std::exception &exc) {
        std::cerr << std::endl
                  << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Exception on processing: " << std::endl
                  << exc.what() << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;

        return 1;
    } catch (...) {
        std::cerr << std::endl
                  << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Unknown exception!" << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        return 1;
    }

    return 0;
}
