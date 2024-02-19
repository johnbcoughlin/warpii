#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/time_stepping.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/vectorization.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/numerics/data_out.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>

#include "RadialEulerOperator.h"
#include "RadialEulerPostprocessor.h"
#include "rk.h"

namespace CylindricalEuler {
using namespace dealii;

constexpr unsigned int testcase = 0;
constexpr unsigned int fe_degree = 5;
constexpr double final_time = 100.0;
constexpr double output_tick = 2.0;

using Number = double;

constexpr double gamma = 5.0 / 3.0;

const double courant_number = 0.15 / std::pow(fe_degree, 1.5);
constexpr LowStorageRungeKuttaScheme lsrk_scheme = stage_5_order_4;

template <int dim> class InitialCondition : public Function<dim> {
public:
  InitialCondition(const unsigned int testcase, const double gamma,
                   const double time)
      : Function<dim>(dim + 2, time), testcase(testcase), gamma(gamma) {}

  virtual double value(const Point<dim> &p,
                       const unsigned int component = 0) const override;

private:
  unsigned int testcase;
  double gamma;
};

template <int dim>
double InitialCondition<dim>::value(const Point<dim> &r,
                                    const unsigned int component) const {
  switch (testcase) {
  case 0: {
    Assert(dim == 1, ExcNotImplemented());

    double density = 1.0 / r[0];
    double u = 0.1;
    double KE = 0.5 * density * u * u;
    double p = 1.0;
    double energy = KE + p / (gamma - 1.0);

    if (component == 0) {
      return r[0] * density;
    } else if (component == 1) {
      return r[0] * density * u;
    } else if (component == 2) {
      return r[0] * energy;
    }
  }

  default:
    Assert(false, ExcNotImplemented());
    return 0.;
  }
}

template<int dim>
class RadialEulerProblem {
public:
  RadialEulerProblem();

  void run();

private:
  double time, time_step;

  ConditionalOStream pcout;
  Triangulation<dim> triangulation;
  FESystem<dim> fe;
  MappingQ<dim> mapping;
  DoFHandler<dim> dof_handler;

  TimerOutput timer;

  RadialEulerOperator<dim, fe_degree, fe_degree + 1> euler_operator;

  void make_grid_and_dofs();

  void output_results(const unsigned int result_number);

  LinearAlgebra::distributed::Vector<Number> solution;

  void project(const Function<1> &function,
               LinearAlgebra::distributed::Vector<Number> &solution) const;
};

template <int dim>
RadialEulerProblem<dim>::RadialEulerProblem()
    : time(0.0), time_step(0.0),
      pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
#ifdef DEAL_II_WITH_P4EST
      ,
      triangulation(MPI_COMM_WORLD)
#endif
      ,
      fe(FE_DGQ<dim>(fe_degree) ^ (dim+2)), mapping(fe_degree),
      dof_handler(triangulation),
      timer(pcout, TimerOutput::never, TimerOutput::wall_times),
      euler_operator(timer) {
}

template <int dim>
void RadialEulerProblem<dim>::make_grid_and_dofs() {
  switch (testcase) {
  case 0: {
    Point<dim> left(1.0);
    Point<dim> right(5.0);
    GridGenerator::hyper_rectangle(triangulation, left, right);
    triangulation.refine_global(2);

    const auto ic = InitialCondition<dim>(0, gamma, 0.0);
    euler_operator.set_inflow_boundary(
        0, std::make_unique<InitialCondition<dim>>(0, gamma, 0.0));

    // std::vector<Number> outflow_vals = {0.0, 0.0, ic.value(right, 2)};
    // euler_operator.set_subsonic_outflow_boundary(
    // 1,
    // std::make_unique<Functions::ConstantFunction<1>>(outflow_vals));
    euler_operator.set_supersonic_outflow_boundary(1);

    break;
  }
  }

  triangulation.refine_global(5);
  dof_handler.distribute_dofs(fe);

  euler_operator.reinit(mapping, dof_handler);
  euler_operator.initialize_vector(solution);
}

void RadialEulerProblem::output_results(const unsigned int result_number) {
  {
    TimerOutput::Scope t(timer, "output");

    RadialEulerPostprocessor postprocessor(gamma);
    DataOut<1> data_out;

    DataOutBase::VtkFlags flags;
    flags.write_higher_order_cells = false;
    data_out.set_flags(flags);

    data_out.attach_dof_handler(dof_handler);
    {
      std::vector<std::string> names;
      names.emplace_back("r_density");
      for (unsigned int d = 0; d < 1; ++d)
        names.emplace_back("r_momentum");
      names.emplace_back("r_energy");

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
    std::ofstream of;
    of.open(filename);
    data_out.write_vtu(of);
    of.close();
  }
}

void RadialEulerProblem::run() {
  {
    const unsigned int n_vect_number = VectorizedArray<Number>::size();
    const unsigned int n_vect_bits = 8 * sizeof(Number) * n_vect_number;

    pcout << "Running with " << Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)
          << " MPI processes" << std::endl;
    pcout << "Vectorization over " << n_vect_number << ' '
          << (std::is_same_v<Number, double> ? "doubles" : "floats") << " = "
          << n_vect_bits << " bits ("
          << Utilities::System::get_current_vectorization_level() << ')'
          << std::endl;
  }

  make_grid_and_dofs();

  const LowStorageRungeKuttaIntegrator<Number> integrator(lsrk_scheme);

  LinearAlgebra::distributed::Vector<Number> rk_register_1;
  LinearAlgebra::distributed::Vector<Number> rk_register_2;
  rk_register_1.reinit(solution);
  rk_register_2.reinit(solution);

  euler_operator.project(InitialCondition<1>(testcase, gamma, 0.0), solution);

  double min_vertex_distance = std::numeric_limits<double>::max();
  for (const auto &cell : triangulation.active_cell_iterators())
    if (cell->is_locally_owned())
      min_vertex_distance =
          std::min(min_vertex_distance, cell->minimum_vertex_distance());
  min_vertex_distance =
      Utilities::MPI::min(min_vertex_distance, MPI_COMM_WORLD);

  double max_time_step = 1e-2;
  time_step = courant_number * integrator.n_stages() /
              euler_operator.compute_cell_transport_speed(solution);
  time_step = std::min(time_step, max_time_step);
  pcout << "Time step size: " << time_step
        << ", minimal h: " << min_vertex_distance
        << ", initial transport scaling: "
        << 1. / euler_operator.compute_cell_transport_speed(solution)
        << std::endl
        << std::endl;

  output_results(0);

  unsigned int timestep_number = 0;

  while (time < final_time - 1e-12) {
    ++timestep_number;
    if (timestep_number % 5 == 0)
      time_step = courant_number * integrator.n_stages() /
                  Utilities::truncate_to_n_digits(
                      euler_operator.compute_cell_transport_speed(solution), 3);

    time_step = std::min(time_step, max_time_step);

    {
      TimerOutput::Scope t(timer, "rk time stepping total");
      integrator.perform_time_step(euler_operator, time, time_step, solution,
                                   rk_register_1, rk_register_2);
    }

    time += time_step;

    if (static_cast<int>(time / output_tick) !=
            static_cast<int>((time - time_step) / output_tick) ||
        time >= final_time - 1e-12)
      output_results(static_cast<unsigned int>(std::round(time / output_tick)));
  }

  timer.print_wall_time_statistics(MPI_COMM_WORLD);
  pcout << std::endl;
}

} // namespace CylindricalEuler

int main(int argc, char **argv) {
  using namespace CylindricalEuler;
  using namespace dealii;

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  try {
    deallog.depth_console(0);

    RadialEulerProblem euler_problem;
    euler_problem.run();
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
