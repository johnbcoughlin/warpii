#pragma once
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/patterns.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <fstream>
#include <memory>

#include "../app.h"
#include "../grid.h"
#include "../wrapper.h"
#include "../dgsem/nodal_dg_discretization.h"
#include "dg_solver.h"
#include "../timestepper.h"
#include "postprocessor.h"
#include "solution_vec.h"
#include "species.h"
#include "extension.h"
#include "dg_solution_helper.h"

using namespace dealii;

namespace warpii {
namespace five_moment {

class FiveMomentWrapper : public ApplicationWrapper {
   public:
    void declare_parameters(ParameterHandler &prm) override;

    std::unique_ptr<AbstractApp> create_app(ParameterHandler &prm,
                                            std::string input,
                                            std::shared_ptr<warpii::Extension> extension) override;
};

template <int dim>
class FiveMomentApp : public AbstractApp {
   public:
    FiveMomentApp(
        std::shared_ptr<five_moment::Extension<dim>> extension,
        std::shared_ptr<NodalDGDiscretization<dim>> discretization,
        std::vector<std::shared_ptr<Species<dim>>> species,
        std::shared_ptr<Grid<dim>> grid,
        std::unique_ptr<FiveMomentDGSolver<dim>> solver, 
        double gas_gamma,
        bool fields_enabled,
        bool write_output,
        double t_end,
        unsigned int n_writeout_frames
        )
        : extension(extension),
          discretization(discretization),
          species(species),
          grid(grid),
          solver(std::move(solver)),
          gas_gamma(gas_gamma),
          fields_enabled(fields_enabled),
          write_output(write_output),
          t_end(t_end),
          n_writeout_frames(n_writeout_frames)
    {}

    static void declare_parameters(ParameterHandler &prm,
            std::shared_ptr<five_moment::Extension<dim>> ext);

    static std::unique_ptr<FiveMomentApp<dim>> create_from_parameters(
        ParameterHandler &prm, std::shared_ptr<five_moment::Extension<dim>> ext);

    void setup(WarpiiOpts opts) override;

    void run(WarpiiOpts opts) override;

    void reinit(ParameterHandler &prm);

    NodalDGDiscretization<dim> &get_discretization();

    FiveMomentDGSolutionHelper<dim> &get_solution_helper();

    FiveMomentDGSolver<dim> &get_solver();

    FiveMSolutionVec &get_solution();

    void output_results(const unsigned int result_number);

   private:
    std::shared_ptr<five_moment::Extension<dim>> extension;
    std::shared_ptr<NodalDGDiscretization<dim>> discretization;
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
void FiveMomentApp<dim>::declare_parameters(ParameterHandler &prm,
        std::shared_ptr<five_moment::Extension<dim>> ext) {
    unsigned int n_species = prm.get_integer("n_species");
    unsigned int n_boundaries = prm.get_integer("n_boundaries");

    PlasmaNormalization::declare_parameters(prm);

    std::vector<Species<dim>> species;
    for (unsigned int i = 0; i < n_species; i++) {
        std::stringstream subsection_name;
        subsection_name << "Species_" << i + 1;
        prm.enter_subsection(subsection_name.str());
        Species<dim>::declare_parameters(prm, n_boundaries);
        prm.leave_subsection();
    }
    PHMaxwellFields<dim>::declare_parameters(prm, n_boundaries);

    Grid<dim>::declare_parameters(prm, ext);

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
    ParameterHandler &prm, 
    std::shared_ptr<five_moment::Extension<dim>> ext) {
    unsigned int n_species = prm.get_integer("n_species");
    unsigned int n_boundaries = prm.get_integer("n_boundaries");

    double gas_gamma = prm.get_double("gas_gamma");

    PlasmaNormalization plasma_norm = PlasmaNormalization::create_from_parameters(prm);

    std::vector<std::shared_ptr<Species<dim>>> species;
    for (unsigned int i = 0; i < n_species; i++) {
        std::stringstream subsection_name;
        subsection_name << "Species_" << i + 1;
        prm.enter_subsection(subsection_name.str());
        species.push_back(
            Species<dim>::create_from_parameters(prm, n_boundaries, gas_gamma));
        prm.leave_subsection();
    }
    auto fields = PHMaxwellFields<dim>::create_from_parameters(
            prm, n_boundaries, plasma_norm);

    auto grid = Grid<dim>::create_from_parameters(prm, 
            std::static_pointer_cast<GridExtension<dim>>(ext));

    unsigned int fe_degree = prm.get_integer("fe_degree");
    std::string fields_enabled_str = prm.get("fields_enabled");
    bool fields_enabled = (fields_enabled_str == "true" || (fields_enabled_str == "auto" && n_species > 1));

    unsigned int n_field_components = fields_enabled ? 8 : 0;
    unsigned int n_components = n_species * 5 + n_field_components;
    double t_end = prm.get_double("t_end");
    bool write_output = prm.get_bool("write_output");
    unsigned int n_writeout_frames = prm.get_integer("n_writeout_frames");

    auto discretization = std::make_shared<NodalDGDiscretization<dim>>(
        grid, n_components, fe_degree);

    auto dg_solver = std::make_unique<FiveMomentDGSolver<dim>>(
        discretization, species, fields, gas_gamma, t_end, n_boundaries, fields_enabled);

    auto app = std::make_unique<FiveMomentApp<dim>>(ext, discretization, species,
                                                    grid, std::move(dg_solver),
                                                    gas_gamma, 
                                                    fields_enabled,
                                                    write_output,
                                                    t_end,
                                                    n_writeout_frames);

    return app;
}

template <int dim>
NodalDGDiscretization<dim> &FiveMomentApp<dim>::get_discretization() {
    return *discretization;
}

template <int dim>
FiveMomentDGSolutionHelper<dim>& FiveMomentApp<dim>::get_solution_helper() {
    return solver->get_solution_helper();
}

template <int dim>
FiveMSolutionVec &FiveMomentApp<dim>::get_solution() {
    return solver->get_solution();
}

template <int dim>
FiveMomentDGSolver<dim> &FiveMomentApp<dim>::get_solver() {
    return *solver;
}

template <int dim>
void FiveMomentApp<dim>::setup(WarpiiOpts) {
    std::cout << "Setting up" << std::endl;
    grid->reinit();
    if (dim == 2) {
        grid->output_svg("grid.svg");
    }
    std::cout << "Wrote out svg" << std::endl;
    solver->reinit();
    solver->project_initial_condition();
    output_results(0);
}

template <int dim>
void FiveMomentApp<dim>::run(WarpiiOpts) {
    double writeout_interval = t_end / n_writeout_frames;
    auto writeout = [&](double t) -> void {
        output_results(static_cast<unsigned int>(std::round(t / writeout_interval)));
    };
    // skip the zeroth writeout because we already did that in the setup phase
    TimestepCallback writeout_callback = TimestepCallback(writeout_interval, writeout, false);

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

            names.emplace_back(sp->name + "_x_momentum");
            names.emplace_back(sp->name + "_y_momentum");
            names.emplace_back(sp->name + "_z_momentum");
            for (unsigned int d = 0; d < 3; ++d) {
                interpretation.push_back(DataComponentInterpretation::component_is_scalar);
            }

            names.emplace_back(sp->name + "_energy");
            interpretation.push_back(
                DataComponentInterpretation::component_is_scalar);
        }
        if (fields_enabled) {
            for (unsigned int d = 0; d < 3; ++d) {
                names.emplace_back("E_field");
                interpretation.push_back(
                    DataComponentInterpretation::component_is_scalar);
            }
            for (unsigned int d = 0; d < 3; ++d) {
                names.emplace_back("B_field");
                interpretation.push_back(
                    DataComponentInterpretation::component_is_scalar);
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
