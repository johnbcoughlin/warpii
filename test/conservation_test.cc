#include <gtest/gtest.h>
#include "src/five_moment/five_moment.h"
#include "src/warpii.h"

using namespace dealii;
using namespace warpii;

TEST(ConservationTest, GlobalIntegralsTest) {
    std::string input = R"(
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
        set VariablesType = Primitive
        set Function constants = pi=3.1415926535
        set Function expression = 1.4 + 0.6 * sin(2*pi*x); 0.98; 1.0
    end
end
    )";

    Warpii warpii_obj;
    warpii_obj.input = input;
    warpii_obj.opts.fpe = true;

    warpii_obj.setup();
    auto& app = warpii_obj.get_app<five_moment::FiveMomentApp<1>>();
    auto& disc = app.get_discretization();
    auto& soln = app.get_solution();

    auto ic_global_integral = disc.compute_global_integral(soln.mesh_sol, 0);
    EXPECT_NEAR(ic_global_integral[0], 1.4, 1e-15);
    EXPECT_NEAR(ic_global_integral[1], 0.98 * 1.4, 1e-15);
}

TEST(ConservationTest, Periodic1D) {
    std::string input = R"(
set Application = FiveMoment
set n_dims = 1
set t_end = 0.04
set write_output = false

set fe_degree = 4

subsection geometry
    set left = 0.0
    set right = 1.0
end

subsection Species_1
    subsection InitialCondition
        set Function constants = pi=3.1415926535
        set Function expression = 1 + 0.6 * sin(2*pi*x); 1.0; 1.0
    end
end
    )";

    Warpii warpii_obj;
    warpii_obj.input = input;
    warpii_obj.opts.fpe = true;

    warpii_obj.setup();
    auto& app = warpii_obj.get_app<five_moment::FiveMomentApp<1>>();
    auto& disc = app.get_discretization();
    auto& soln = app.get_solution();

    auto ic_global_integral = disc.compute_global_integral(soln.mesh_sol, 0);

    warpii_obj.run();
    auto global_integral = disc.compute_global_integral(soln.mesh_sol, 0);
    for (unsigned int comp = 0; comp < 3; comp++) {
        EXPECT_NEAR(global_integral[comp], ic_global_integral[comp], 1e-13);
    }
}

TEST(ConservationTest, NonPeriodic1D) {
    std::string input = R"(
set Application = FiveMoment
set n_dims = 1
set t_end = 0.04
set write_output = false

set fe_degree = 4
set n_boundaries = 2

subsection geometry
    set left = 0.0
    set right = 1.0
    set periodic_dimensions =
end

subsection Species_1
    subsection InitialCondition
        set Function constants = pi=3.1415926535
        set Function expression = 1 + 0.6 * sin(2*pi*x); 1.0; 1.0
    end

    subsection BoundaryConditions
        set 0 = Inflow
        subsection 0_Inflow
            set Function constants = gamma=1.66667,rhoL=3.857,uL=2.629,pL=10.333
            set Function expression = rhoL; uL; pL
        end
        set 1 = Outflow
    end
end
    )";

    Warpii warpii_obj;
    warpii_obj.input = input;
    warpii_obj.opts.fpe = true;

    warpii_obj.setup();
    auto& app = warpii_obj.get_app<five_moment::FiveMomentApp<1>>();
    auto& disc = app.get_discretization();
    auto& soln = app.get_solution();

    auto ic_global_integral = disc.compute_global_integral(soln.mesh_sol, 0);

    warpii_obj.run();
    auto global_integral = disc.compute_global_integral(soln.mesh_sol, 0);
    auto inflow_flux = soln.boundary_integrated_fluxes.at_boundary<1>(0);
    auto outflow_flux = soln.boundary_integrated_fluxes.at_boundary<1>(1);
    auto balance = global_integral + inflow_flux + outflow_flux;
    for (unsigned int comp = 0; comp < 3; comp++) {
        EXPECT_NEAR(balance[comp], ic_global_integral[comp], 1e-15);
    }
}
