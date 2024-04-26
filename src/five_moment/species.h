#pragma once

#include <deal.II/base/function_parser.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/parsed_function.h>
#include <deal.II/base/patterns.h>
#include <deal.II/base/types.h>
#include <deal.II/base/utilities.h>

#include <sstream>
#include <string>

#include "bc_helper.h"

using namespace dealii;

namespace warpii {
namespace five_moment {

const unsigned int MAX_BOUNDARY_CONDITIONS = 8;

template <int dim>
class Species {
   public:
    Species(std::string name, double charge, double mass,
            EulerBCMap<dim> bc_map, std::unique_ptr<Functions::ParsedFunction<dim>> initial_condition)
        : name(name),
          charge(charge),
          mass(mass),
          bc_map(bc_map),
          initial_condition(std::move(initial_condition)) {}

    static void declare_parameters(ParameterHandler &prm,
                                   unsigned int n_boundaries);

    static std::shared_ptr<Species<dim>> create_from_parameters(
        ParameterHandler &prm, unsigned int n_boundaries);

   private:

   public:
    std::string name;
    double charge;
    double mass;
    EulerBCMap<dim> bc_map;
    std::unique_ptr<Functions::ParsedFunction<dim>> initial_condition;
};

template <int dim>
void Species<dim>::declare_parameters(ParameterHandler &prm,
                                      unsigned int n_boundaries) {
    prm.declare_entry("name", "neutral",
                      Patterns::Selection("neutral|ion|electron"));
    prm.declare_entry("charge", "0.0", Patterns::Double());
    prm.declare_entry("mass", "1.0", Patterns::Double(0.0));
    prm.enter_subsection("BoundaryConditions");
    {
        for (unsigned int i = 0; i < n_boundaries; i++) {
            prm.declare_entry(std::to_string(i), "Wall",
                              Patterns::Selection("Wall|Outflow|Inflow"));
            std::stringstream ss;
            ss << i << "_Inflow";
            prm.enter_subsection(ss.str());
            Functions::ParsedFunction<dim>::declare_parameters(prm, dim + 2);
            prm.leave_subsection();
        }
    }
    prm.leave_subsection();  // BoundaryConditions
    prm.enter_subsection("InitialCondition");
    Functions::ParsedFunction<dim>::declare_parameters(prm, dim + 2);
    prm.leave_subsection();
}

template <int dim>
std::shared_ptr<Species<dim>> Species<dim>::create_from_parameters(
    ParameterHandler &prm, unsigned int n_boundaries) {
    std::string name = prm.get("name");
    double charge = prm.get_double("charge");
    double mass = prm.get_double("mass");
    auto bc_map = EulerBCMap<dim>();

    prm.enter_subsection("BoundaryConditions");
    {
        for (unsigned int i = 0; i < n_boundaries; i++) {
            std::string bc_type = prm.get(std::to_string(i));
            auto boundary_id = static_cast<types::boundary_id>(i);
            if (bc_type == "Wall") {
                bc_map.set_wall_boundary(boundary_id);
            } else if (bc_type == "Outflow") {
                bc_map.set_supersonic_outflow_boundary(boundary_id);
            } else if (bc_type == "Inflow") {
                std::stringstream ss;
                ss << i << "_Inflow";
                prm.enter_subsection(ss.str());
                std::unique_ptr<Functions::ParsedFunction<dim>> inflow_func =
                    std::make_unique<Functions::ParsedFunction<dim>>(dim + 2);
                inflow_func->parse_parameters(prm);
                bc_map.set_inflow_boundary(boundary_id, std::move(inflow_func));
                prm.leave_subsection();
            }
        }
    }
    prm.leave_subsection();
    prm.enter_subsection("InitialCondition");
    std::unique_ptr<Functions::ParsedFunction<dim>> initial_condition =
        std::make_unique<Functions::ParsedFunction<dim>>(dim + 2);
    initial_condition->parse_parameters(prm);
    prm.leave_subsection();

    return std::make_shared<Species<dim>>(name, charge, mass, bc_map,
                                          std::move(initial_condition));
}

}  // namespace five_moment
}  // namespace warpii
