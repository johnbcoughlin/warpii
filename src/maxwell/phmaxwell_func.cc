#include "phmaxwell_func.h"

#include <deal.II/base/parsed_function.h>
#include <deal.II/base/parameter_handler.h>

using namespace dealii;

namespace warpii {

template <int dim>
void PHMaxwellFunc<dim>::declare_parameters(ParameterHandler& prm) {
    prm.enter_subsection("E field");
    Functions::ParsedFunction<dim>::declare_parameters(prm, 3);
    prm.leave_subsection();
    prm.enter_subsection("B field");
    Functions::ParsedFunction<dim>::declare_parameters(prm, 3);
    prm.leave_subsection();
    prm.enter_subsection("phi");
    Functions::ParsedFunction<dim>::declare_parameters(prm, 1);
    prm.leave_subsection();
    prm.enter_subsection("psi");
    Functions::ParsedFunction<dim>::declare_parameters(prm, 1);
    prm.leave_subsection();
}

template <int dim>
PHMaxwellFunc<dim> PHMaxwellFunc<dim>::create_from_parameters(ParameterHandler &prm) {
    auto E_func = std::make_unique<Functions::ParsedFunction<dim>>(3);
    auto B_func = std::make_unique<Functions::ParsedFunction<dim>>(3);
    auto phi_func = std::make_unique<Functions::ParsedFunction<dim>>(1);
    auto psi_func = std::make_unique<Functions::ParsedFunction<dim>>(1);

    prm.enter_subsection("E field");
    E_func.parse_parameters(prm);
    prm.leave_subsection();

    prm.enter_subsection("B field");
    B_func.parse_parameters(prm);
    prm.leave_subsection();

    prm.enter_subsection("phi");
    phi_func.parse_parameters(prm);
    prm.leave_subsection();

    prm.enter_subsection("psi");
    psi_func.parse_parameters(prm);
    prm.leave_subsection();

    return PHMaxwellFunc(
            std::move(E_func),
            std::move(B_func),
            std::move(phi_func),
            std::move(psi_func));
}

}  // namespace warpii
