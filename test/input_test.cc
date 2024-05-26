#include <deal.II/base/function_parser.h>
#include <gtest/gtest.h>
#include "deal.II/base/parameter_handler.h"
#include "src/five_moment/five_moment.h"
#include "src/warpii.h"

using namespace dealii;
using namespace warpii;

TEST(InputTest, DefaultInputIsValid) {
    Warpii warpii_obj;
    warpii_obj.input = R"(
set write_output = false
    )";
    //warpii_obj.run();
}

TEST(InputTest, FreeStream1D) {
    std::string input_template = R"(
set Application = FiveMoment
set n_dims = 1
set t_end = 0.04
set write_output = false

set fe_degree = 2

subsection geometry
    set left = 0.0
    set right = 1.0
end

subsection Species_1
    subsection InitialCondition
        set Function constants = pi=3.1415926535
        set Function expression = 1 + 0.6 * sin(2*pi*x);\
                                  1 + 0.6 * sin(2*pi*x);\
                                  0.5 * (1 + 0.6*sin(2*pi*x)) + 1.5
    end
end
    )";

    // Run for a short time to avoid long time integration error messing
    // with convergence order.
    FunctionParser<1> expected_density = FunctionParser<1>(
            "1 + 0.6 * sin(2*pi*(x - 0.04)); 0; 0", "pi=3.1415926535");

    std::vector<unsigned int> Nxs = { 20, 30 };
    std::vector<double> errors;
    for (unsigned int i = 0; i < Nxs.size(); i++) {
        Warpii warpii_obj;
        std::stringstream input;
        input << input_template;
        input << "subsection geometry\n set nx = " << Nxs[i] << "\n end";
        warpii_obj.opts.fpe = true;
        warpii_obj.input = input.str();
        warpii_obj.run();
        auto& app = warpii_obj.get_app<five_moment::FiveMomentApp<1>>();
        auto& disc = app.get_discretization();
        auto& soln = app.get_solution();

        double error = disc.compute_global_error(soln.mesh_sol, expected_density, 0);
        std::cout << "error = " << error << std::endl;
        errors.push_back(error);
    }
    EXPECT_NEAR(errors[1], 0.0, 1e-4);
    EXPECT_NEAR(errors[0] / errors[1], pow(30.0/20.0, 3), 1.0);
}

TEST(InputTest, SodShocktube) {
    Warpii warpii_obj;
    std::string input = R"(
set Application = FiveMoment
set n_dims = 1
set t_end = 0.1
set write_output = false

set fe_degree = 4

set n_boundaries = 2

subsection geometry
    set left = 0.0
    set right = 1.0
    set nx = 100
    set periodic_dimensions =
end

subsection Species_1
    subsection InitialCondition
        set Function constants = gamma=1.66667
        set Function expression = if(x < 0.5, 1.0, 0.10); \
                                  0.0; \
                                  if(x < 0.5, 1.0, 0.125) / (gamma - 1)
    end

    subsection BoundaryConditions
        set 0 = Outflow
        set 1 = Outflow
    end
end
    )";

    warpii_obj.opts.fpe = true;
    warpii_obj.input = input;
    warpii_obj.run();
}

TEST(InputTest, FreeStreamPseudo2D) {
    Warpii warpii_obj;
    std::string input = R"(
set Application = FiveMoment
set n_dims = 2
set t_end = 0.1
set write_output = false

set fe_degree = 2

subsection geometry
    set left = 0.0,0.0
    set right = 1.0,0.02
    set nx = 100,2
end

subsection Species_1
    subsection InitialCondition
        set Function constants = pi=3.1415926535
        set Function expression = 1 + 0.6 * sin(2*pi*x); \
                                  1 + 0.6 * sin(2*pi*x); \
                                  0.0; \
                                  0.5 * (1 + 0.6*sin(2*pi*x)) + 1.5
    end
end
    )";

    warpii_obj.opts.fpe = true;
    warpii_obj.input = input;
    warpii_obj.run();
}
