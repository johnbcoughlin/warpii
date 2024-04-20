#include <gtest/gtest.h>
#include "deal.II/base/parameter_handler.h"
#include "src/warpii.h"

using namespace dealii;
using namespace warpii;

TEST(InputTest, Grid) {
    Warpii warpii_obj;
    warpii_obj.input = R"(
set Application = FiveMoment
set n_dims = 1
subsection geometry
    set nx = 4
end
    )";
    warpii_obj.run();
}
