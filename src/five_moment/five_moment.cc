#include "five_moment.h"

#include <deal.II/base/exceptions.h>
#include <deal.II/base/utilities.h>

#include <memory>

#include "../wrapper.h"

namespace warpii {
namespace five_moment {

void FiveMomentWrapper::declare_parameters(ParameterHandler &prm) {
    prm.declare_entry("n_dims", "1", Patterns::Integer(1, 3), 
R"(The number of dimensions in the problem.
    )");
    prm.declare_entry("n_species", "1", Patterns::Integer());
    prm.declare_entry("n_boundaries", "1", Patterns::Integer());
}

std::unique_ptr<AbstractApp> FiveMomentWrapper::create_app(
    ParameterHandler &prm, std::string input) {
    prm.parse_input_from_string(input, "", true);

    switch (prm.get_integer("n_dims")) {
        case 1: {
            FiveMomentApp<1>::declare_parameters(prm);
            prm.parse_input_from_string(input, "", false);
            return FiveMomentApp<1>::create_from_parameters(prm);
        }
        case 2: {
            FiveMomentApp<2>::declare_parameters(prm);
            prm.parse_input_from_string(input, "", false);
            return FiveMomentApp<2>::create_from_parameters(prm);
        }
        case 3: {
            FiveMomentApp<3>::declare_parameters(prm);
            prm.parse_input_from_string(input, "", false);
            return FiveMomentApp<3>::create_from_parameters(prm);
        }
        default: {
            AssertThrow(false, ExcMessage("n_dims must be 1, 2, or 3"));
        }
    }
}

}  // namespace five_moment
}  // namespace warpii
