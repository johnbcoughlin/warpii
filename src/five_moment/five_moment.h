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
#include "dg_operator.h"
#include "dg_solution.h"
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

    std::unique_ptr<AbstractApp> create_app(ParameterHandler &prm, std::string input) override;

   private:
    std::unique_ptr<AbstractFiveMomentApp> app;
};

template <int dim>
class FiveMomentApp : public AbstractApp {
   public:
    FiveMomentApp(std::vector<Species<dim>> species,
                  std::unique_ptr<FiveMomentDGSolution<dim>> solution,
                  FiveMomentDGOperator<dim> dg_operator)
        : species(species),
          solution(std::move(solution)),
          dg_operator(dg_operator) {}

    static void declare_parameters(ParameterHandler &prm);

    static std::unique_ptr<FiveMomentApp<dim>> create_from_parameters(
        ParameterHandler &prm);

    void run(WarpiiOpts opts) override;

    void reinit(ParameterHandler &prm);

   private:
    std::vector<Species<dim>> species;
    std::unique_ptr<FiveMomentDGSolution<dim>> solution;
    FiveMomentDGOperator<dim> dg_operator;
};

template <int dim>
void FiveMomentApp<dim>::declare_parameters(
        ParameterHandler &prm) {
    unsigned int n_species = prm.get_integer("n_species");
    unsigned int n_boundaries = prm.get_integer("n_boundaries");

    std::vector<Species<dim>> species;
    for (unsigned int i = 0; i < n_species; i++) {
        std::stringstream subsection_name;
        subsection_name << "Species_" << i;
        prm.enter_subsection(subsection_name.str());
        Species<dim>::declare_parameters(prm, n_boundaries);
        prm.leave_subsection();
    }

    Grid<dim>::declare_parameters(prm);

    prm.declare_entry("fe_degree", "2", Patterns::Integer(1, 6));
    prm.declare_entry("fields_enabled", "true", Patterns::Bool());

    //prm.print_parameters(std::cout, ParameterHandler::OutputStyle::ShortPRM);
}

template <int dim>
std::unique_ptr<FiveMomentApp<dim>> FiveMomentApp<dim>::create_from_parameters(
    ParameterHandler &prm) {
    unsigned int n_species = prm.get_integer("n_species");

    std::vector<Species<dim>> species;
    for (unsigned int i = 0; i < n_species; i++) {
        std::stringstream subsection_name;
        subsection_name << "Species_" << i;
        prm.enter_subsection(subsection_name.str());
        species.push_back(Species<dim>::create_from_parameters(prm));
        prm.leave_subsection();
    }

    prm.print_parameters(std::cout, ParameterHandler::OutputStyle::ShortPRM);
    auto grid = Grid<dim>::create_from_parameters(prm);

    unsigned int fe_degree = prm.get_integer("fe_degree");
    bool fields_enabled = prm.get_bool("fields_enabled");
    unsigned int n_field_components = fields_enabled ? 6 : 0;
    unsigned int n_components = n_species * (dim + 2) + n_field_components;

    auto dg_solution = std::make_unique<FiveMomentDGSolution<dim>>(
        grid, n_components, fe_degree);
    auto dg_operator = FiveMomentDGOperator<dim>(species);

    auto app = std::make_unique<FiveMomentApp<dim>>(
        species, std::move(dg_solution), dg_operator);

    return app;
}

template <int dim>
void FiveMomentApp<dim>::run(WarpiiOpts ) {
}

}  // namespace five_moment
}  // namespace warpii
