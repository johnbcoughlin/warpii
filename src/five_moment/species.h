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

}  // namespace five_moment
}  // namespace warpii
