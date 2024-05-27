#include "species.h"
#include "species_func.h"
#include <deal.II/base/patterns.h>

namespace warpii {
namespace five_moment {

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
    SpeciesFunc<dim>::declare_parameters(prm);
    prm.leave_subsection();
}

template <int dim>
std::shared_ptr<Species<dim>> Species<dim>::create_from_parameters(
    ParameterHandler &prm, unsigned int n_boundaries, double gas_gamma) {
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
    SpeciesFunc<dim> initial_condition = SpeciesFunc<dim>::create_from_parameters(prm, gas_gamma);
    prm.leave_subsection();

    return std::make_shared<Species<dim>>(name, charge, mass, bc_map,
                                          std::move(initial_condition));
}

template class Species<1>;
template class Species<2>;

}  // namespace five_moment

}  // namespace warpii
