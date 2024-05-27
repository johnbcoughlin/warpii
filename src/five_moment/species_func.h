#pragma once
#include <deal.II/base/function_parser.h>
#include <deal.II/base/parsed_function.h>

using namespace dealii;

namespace warpii {
namespace five_moment {

enum SpeciesFuncVariablesType {
    CONSERVED,
    PRIMITIVE,
};

template <int dim>
class SpeciesFunc : public Function<dim> {
   public:
    SpeciesFunc(std::unique_ptr<Functions::ParsedFunction<dim>> func,
                SpeciesFuncVariablesType variables_type, double gas_gamma)
        : Function<dim>(dim + 2),
          func(std::move(func)),
          variables_type(variables_type),
          gas_gamma(gas_gamma) {}

    double value(const Point<dim> &pt,
                 const unsigned int component) const override;

    static void declare_parameters(ParameterHandler &prm);

    static SpeciesFunc<dim> create_from_parameters(ParameterHandler &prm,
                                                   double gas_gamma);

   private:
    std::unique_ptr<Functions::ParsedFunction<dim>> func;
    SpeciesFuncVariablesType variables_type;
    double gas_gamma;
};
}  // namespace five_moment
}  // namespace warpii
