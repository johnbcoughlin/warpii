#pragma once
#include <deal.II/base/parsed_function.h>

using namespace dealii;

namespace warpii {

template <int dim>
class PHMaxwellFunc {
   public:
    PHMaxwellFunc(std::unique_ptr<Functions::ParsedFunction<dim>> E_func,
                  std::unique_ptr<Functions::ParsedFunction<dim>> B_func,
                  std::unique_ptr<Functions::ParsedFunction<dim>> phi_func,
                  std::unique_ptr<Functions::ParsedFunction<dim>> psi_func):
        E_func(E_func), B_func(B_func),
        phi_func(phi_func), psi_func(psi_func)
    {}

    static void declare_parameters(ParameterHandler &prm);
    static PHMaxwellFunc<dim> create_from_parameters(ParameterHandler &prm);

    std::unique_ptr<Functions::ParsedFunction<dim>> E_func;
    std::unique_ptr<Functions::ParsedFunction<dim>> B_func;
    std::unique_ptr<Functions::ParsedFunction<dim>> phi_func;
    std::unique_ptr<Functions::ParsedFunction<dim>> psi_func;
};

}  // namespace warpii
