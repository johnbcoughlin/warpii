#include <gtest/gtest.h>
#include "src/five_moment/five_moment.h"
#include "src/warpii.h"

using namespace dealii;
using namespace warpii;

TEST(ShockCapturingFVTest, SingleCell) {
    //char** argv = nullptr;
    //int argc = 0;
    //Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

    Warpii warpii_obj;
    warpii_obj.input = R"(
set Application = FiveMoment
set n_dims = 2

set t_end = 0.01
set fields_enabled = false

set fe_degree = 2

subsection geometry
    set left = 0.0,0.0
    set right = 1.0,1.0
    set nx = 1,1
end
subsection Species_1
    subsection InitialCondition
        set Function constants = pi=3.1415926535
        set Function expression = 1 + 0.6 * sin(2*pi*(x+y)); \
                                  1 + 0.6 * sin(2*pi*(x+y)); \
                                  1 + 0.6 * sin(2*pi*(x+y)); \
                                  0.5 * 2*(1 + 0.6*sin(2*pi*(x+y))) + 1.5
    end
end
)";
    warpii_obj.run();
}
