#include "dg_solver.h"

namespace warpii {
namespace five_moment {

template <int dim>
void FiveMomentDGSolver<dim>::reinit() {
    discretization->reinit();
    discretization->perform_allocation(solution.mesh_sol);
    solution.boundary_integrated_fluxes.reinit(n_boundaries, dim);
    ssp_integrator.reinit(solution, 3);
}

template <int dim>
void FiveMomentDGSolver<dim>::project_initial_condition() {
    for (unsigned int i = 0; i < species.size(); i++) {
        solution_helper.project_fluid_quantities(
            *species.at(i)->initial_condition, solution.mesh_sol, i);
    }
}

template <int dim>
void FiveMomentDGSolver<dim>::solve(TimestepCallback writeout_callback) {
    auto step = [&](double t, double dt) -> bool {
        ssp_integrator.evolve_one_time_step(flux_operator, solution, dt, t);

        std::cout << "t = " << t << std::endl;
        return true;
    };
    auto recommend_dt = [&]() -> double {
        return fluid_flux_operator.recommend_dt(
            discretization->get_matrix_free(), solution);
    };

    std::vector<TimestepCallback> callbacks = {writeout_callback};
    advance(step, t_end, recommend_dt, callbacks);
}

template <int dim>
FiveMSolutionVec& FiveMomentDGSolver<dim>::get_solution() {
    return solution;
}

template <int dim>
FiveMomentDGSolutionHelper<dim>& FiveMomentDGSolver<dim>::get_solution_helper() {
    return solution_helper;
}

template class FiveMomentDGSolver<1>;
template class FiveMomentDGSolver<2>;

}  // namespace five_moment
}  // namespace warpii
