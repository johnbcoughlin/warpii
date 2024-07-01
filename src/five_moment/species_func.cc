#include "species_func.h"
#include <deal.II/base/parameter_handler.h>
#include <variant>

namespace warpii {
namespace five_moment {

template <int dim>
double SpeciesFunc<dim>::value(const Point<dim> &pt,
                               const unsigned int component) const {
    if (variables_type == CONSERVED) {
        return func->value(pt, component);
    } else {
        double rho = func->value(pt, 0);
        if (component == 0) {
            return rho;
        } else if (component <= 3) {
            return rho * func->value(pt, component);
        } else {
            double kinetic_energy = 0.0;
            for (unsigned int d = 0; d < 3; d++) {
                double u_d = func->value(pt, d + 1);
                kinetic_energy += 0.5 * rho * u_d * u_d;
            }
            double p = func->value(pt, 4);
            return kinetic_energy + p / (gas_gamma - 1);
        }
    }
}

template <int dim>
void SpeciesFunc<dim>::declare_parameters(ParameterHandler& prm) {
    prm.declare_entry("VariablesType", "Primitive", Patterns::Selection("Primitive|Conserved"));
    Functions::ParsedFunction<dim>::declare_parameters(prm, 5);
}

template <int dim>
std::unique_ptr<SpeciesFunc<dim>> SpeciesFunc<dim>::create_from_parameters(ParameterHandler& prm, double gas_gamma) {
    std::string str = prm.get("VariablesType");
    SpeciesFuncVariablesType variables_type;
    if (str == "Primitive") {
        variables_type = PRIMITIVE;
    } else {
        variables_type = CONSERVED;
    }
    std::unique_ptr<Functions::ParsedFunction<dim>> func =
        std::make_unique<Functions::ParsedFunction<dim>>(5);
    func->parse_parameters(prm);
    return std::make_unique<SpeciesFunc<dim>>(std::move(func), variables_type, gas_gamma);
}

template class SpeciesFunc<1>;
template class SpeciesFunc<2>;

}  // namespace five_moment
}  // namespace warpii
