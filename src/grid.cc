#include "grid.h"

#include <deal.II/base/exceptions.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/utilities.h>
#include <deal.II/grid/grid_generator.h>

#include <memory>

namespace warpii {
using namespace dealii;

void GridWrapper::declare_parameters(ParameterHandler& prm) {
    prm.enter_subsection("geometry");
    prm.declare_entry("GridType", "HyperRectangle",
                      Patterns::Selection("HyperRectangle|ForwardFacingStep"),
                      "The type of GridDescription whose parameters are "
                      "supplied in this section.");
    prm.leave_subsection();
}

}  // namespace warpii
