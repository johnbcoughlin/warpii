#pragma once
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/patterns.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <fstream>

#include "../app.h"
#include "../grid.h"
#include "../wrapper.h"
#include "dg_solver.h"
#include "../timestepper.h"
#include "postprocessor.h"
#include "solution_vec.h"
#include "species.h"

using namespace dealii;

namespace warpii {
namespace five_moment {

const unsigned int MAX_N_SPECIES = 8;

class AbstractFiveMomentApp {
   public:
    virtual ~AbstractFiveMomentApp() = default;
};

class FiveMomentWrapper : public ApplicationWrapper {
   public:
    void declare_parameters(ParameterHandler &prm) override;

    std::unique_ptr<AbstractApp> create_app(ParameterHandler &prm,
                                            std::string input) override;

   private:
    std::unique_ptr<AbstractFiveMomentApp> app;
};

template <int dim>
class FiveMomentApp : public AbstractApp {
   public:
    FiveMomentApp(
        std::shared_ptr<FiveMomentDGDiscretization<dim>> discretization,
        std::vector<std::shared_ptr<Species<dim>>> species,
        std::shared_ptr<Grid<dim>> grid,
        std::unique_ptr<FiveMomentDGSolver<dim>> solver, 
        double gas_gamma,
        bool fields_enabled,
        bool write_output,
        double t_end,
        unsigned int n_writeout_frames
        )
        : discretization(discretization),
          species(species),
          grid(grid),
          solver(std::move(solver)),
          gas_gamma(gas_gamma),
          fields_enabled(fields_enabled),
          write_output(write_output),
          t_end(t_end),
          n_writeout_frames(n_writeout_frames)
    {}

    static void declare_parameters(ParameterHandler &prm);

    static std::unique_ptr<FiveMomentApp<dim>> create_from_parameters(
        ParameterHandler &prm);

    void setup(WarpiiOpts opts) override;

    void run(WarpiiOpts opts) override;

    void reinit(ParameterHandler &prm);

    FiveMomentDGDiscretization<dim> &get_discretization();

    FiveMSolutionVec &get_solution();

    void output_results(const unsigned int result_number);

   private:
    std::shared_ptr<FiveMomentDGDiscretization<dim>> discretization;
    std::vector<std::shared_ptr<Species<dim>>> species;
    std::shared_ptr<Grid<dim>> grid;
    std::unique_ptr<FiveMomentDGSolver<dim>> solver;
    double gas_gamma;
    bool fields_enabled;
    bool write_output;
    double t_end;
    unsigned int n_writeout_frames;
};

template <int dim>
void FiveMomentApp<dim>::declare_parameters(ParameterHandler &prm) {
    unsigned int n_species = prm.get_integer("n_species");
    unsigned int n_boundaries = prm.get_integer("n_boundaries");

    std::vector<Species<dim>> species;
    for (unsigned int i = 0; i < n_species; i++) {
        std::stringstream subsection_name;
        subsection_name << "Species_" << i + 1;
        prm.enter_subsection(subsection_name.str());
        Species<dim>::declare_parameters(prm, n_boundaries);
        prm.leave_subsection();
    }

    Grid<dim>::declare_parameters(prm);

    prm.declare_entry("fe_degree", "2", Patterns::Integer(1, 6),
            R"(
The degree of finite element shape functions to use.
The expected order of convergence is one greater than this.
I.e. if fe_degree == 2, then we use quadratic polynomials and
can expect third order convergence.
            )");
    prm.declare_entry("fields_enabled", "auto", Patterns::Selection("true|false|auto"),
            R"(
Whether electromagnetic fields are enabled for this problem.
Values:
    - true: fields are enabled and will be evolved
    - false: fields are disabled
    - auto: fields are enabled if and only if n_species >= 2

If enabled, the solver always uses 8 components for the EM fields regardless of n_dims.
The components are

    [ Ex, Ey, Ez, Bx, By, Bz, phi, psi ],

where phi and psi are the scalar divergence error indicators for Gauss's law and the div-B law,
as used in the perfectly hyperbolic Maxwell's equation system.
            )");
    prm.declare_entry("gas_gamma", "1.6666666666667", Patterns::Double(),
            R"(
The gas gamma, AKA the ratio of specific heats, AKA (n_dims+2)/2 for a plasma.
Defaults to 5/3, the value for simple ions with 3 degrees of freedom.
            )");
    prm.declare_entry("t_end", "0.0", Patterns::Double(0.0));
    prm.declare_entry("write_output", "true", Patterns::Bool());
    prm.declare_entry("n_writeout_frames", "10", Patterns::Integer(0));
}

template <int dim>
std::unique_ptr<FiveMomentApp<dim>> FiveMomentApp<dim>::create_from_parameters(
    ParameterHandler &prm) {
    unsigned int n_species = prm.get_integer("n_species");
    unsigned int n_boundaries = prm.get_integer("n_boundaries");

    std::vector<std::shared_ptr<Species<dim>>> species;
    for (unsigned int i = 0; i < n_species; i++) {
        std::stringstream subsection_name;
        subsection_name << "Species_" << i + 1;
        prm.enter_subsection(subsection_name.str());
        species.push_back(
            Species<dim>::create_from_parameters(prm, n_boundaries));
        prm.leave_subsection();
    }

    auto grid = Grid<dim>::create_from_parameters(prm);

    unsigned int fe_degree = prm.get_integer("fe_degree");
    std::string fields_enabled_str = prm.get("fields_enabled");
    bool fields_enabled = (fields_enabled_str == "true" || (fields_enabled_str == "auto" && n_species > 1));

    unsigned int n_field_components = fields_enabled ? 8 : 0;
    unsigned int n_components = n_species * (dim + 2) + n_field_components;
    double gas_gamma = prm.get_double("gas_gamma");
    double t_end = prm.get_double("t_end");
    bool write_output = prm.get_bool("write_output");
    unsigned int n_writeout_frames = prm.get_integer("n_writeout_frames");

    unsigned int n_nonmesh_unknowns = n_boundaries;

    auto discretization = std::make_shared<FiveMomentDGDiscretization<dim>>(
        grid, n_components, fe_degree);

    auto dg_solver = std::make_unique<FiveMomentDGSolver<dim>>(
        discretization, species, gas_gamma, t_end, n_nonmesh_unknowns);

    auto app = std::make_unique<FiveMomentApp<dim>>(discretization, species,
                                                    grid, std::move(dg_solver),
                                                    gas_gamma, 
                                                    fields_enabled,
                                                    write_output,
                                                    t_end,
                                                    n_writeout_frames);

    return app;
}

template <int dim>
FiveMomentDGDiscretization<dim> &FiveMomentApp<dim>::get_discretization() {
    return *discretization;
}

template <int dim>
FiveMSolutionVec &FiveMomentApp<dim>::get_solution() {
    return solver->get_solution();
}

template <int dim>
void FiveMomentApp<dim>::setup(WarpiiOpts) {
    grid->reinit();
    solver->reinit();
    solver->project_initial_condition();
}

template <int dim>
void FiveMomentApp<dim>::run(WarpiiOpts) {
    double writeout_interval = t_end / n_writeout_frames;
    auto writeout = [&](double t) -> void {
        output_results(static_cast<unsigned int>(std::round(t / writeout_interval)));
    };
    TimestepCallback writeout_callback = TimestepCallback(writeout_interval, writeout);

    solver->solve(writeout_callback);
}

template <int dim>
void FiveMomentApp<dim>::output_results(const unsigned int result_number) {
    FiveMomentPostprocessor<dim> postprocessor(gas_gamma);
    if (!write_output) {
        return;
    }

    DataOut<dim> data_out;

    DataOutBase::VtkFlags flags;
    flags.write_higher_order_cells = false;
    data_out.set_flags(flags);

    auto& sol = get_solution();
    data_out.attach_dof_handler(discretization->get_dof_handler());
    {
        std::vector<std::string> names;
        std::vector<DataComponentInterpretation::DataComponentInterpretation>
            interpretation;

        for (unsigned int i = 0; i < species.size(); i++) {
            auto &sp = species.at(i);
            names.emplace_back(sp->name + "_density");
            interpretation.push_back(
                DataComponentInterpretation::component_is_scalar);

            for (unsigned int d = 0; d < dim; ++d) {
                names.emplace_back(sp->name + "_momentum");
                interpretation.push_back(
                    DataComponentInterpretation::component_is_part_of_vector);
            }

            names.emplace_back(sp->name + "_energy");
            interpretation.push_back(
                DataComponentInterpretation::component_is_scalar);
        }
        if (fields_enabled) {
            for (unsigned int d = 0; d < 3; ++d) {
                names.emplace_back("E_field");
                interpretation.push_back(
                    DataComponentInterpretation::component_is_part_of_vector);
            }
            for (unsigned int d = 0; d < 3; ++d) {
                names.emplace_back("B_field");
                interpretation.push_back(
                    DataComponentInterpretation::component_is_part_of_vector);
            }
            names.emplace_back("ph_maxwell_gauss_error");
            interpretation.push_back(
                DataComponentInterpretation::component_is_scalar);
            names.emplace_back("ph_maxwell_monopole_error");
            interpretation.push_back(
                DataComponentInterpretation::component_is_scalar);
        }

        data_out.add_data_vector(discretization->get_dof_handler(),
                                 sol.mesh_sol, names, interpretation);
    }
    data_out.add_data_vector(sol.mesh_sol, postprocessor);

    Vector<double> mpi_owner(grid->triangulation.n_active_cells());
    mpi_owner = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    data_out.add_data_vector(mpi_owner, "owner");

    discretization->build_data_out_patches(data_out);

    const std::string filename =
        "solution_" + Utilities::int_to_string(result_number, 3) + ".vtu";
    data_out.write_vtu_in_parallel(filename, MPI_COMM_WORLD);
}

}  // namespace five_moment
}  // namespace warpii
