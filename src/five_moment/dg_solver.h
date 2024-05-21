#pragma once

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>

#include <functional>

#include "../grid.h"
#include "../rk.h"
#include "../timestepper.h"
#include "solution_vec.h"
#include "dg_discretization.h"
#include "fluid_flux_dg_operator.h"
#include "fluid_flux_es_dgsem_operator.h"
#include "species.h"

using namespace dealii;

namespace warpii {
namespace five_moment {

/**
 * The DGSolver for the Five-Moment application.
 *
 * This class's job is to perform the whole time integration of the five-moment
 * system of equations, using the supplied DGOperator to calculate right-hand
 * sides.
 */
template <int dim>
class FiveMomentDGSolver {
   public:
    FiveMomentDGSolver(
        std::shared_ptr<FiveMomentDGDiscretization<dim>> discretization,
        std::vector<std::shared_ptr<Species<dim>>> species, double gas_gamma,
        double t_end,
        unsigned int n_nonmesh_unknowns)
        : t_end(t_end),
          discretization(discretization),
          species(species),
          fluid_flux_operator(discretization, gas_gamma, species),
          n_nonmesh_unknowns(n_nonmesh_unknowns)
        {}

    void reinit();

    void solve(TimestepCallback callback);

    LinearAlgebra::distributed::Vector<double>& get_solution();

   private:
    double t_end;
    std::shared_ptr<FiveMomentDGDiscretization<dim>> discretization;
    std::vector<std::shared_ptr<Species<dim>>> species;
    FiveMSolutionVec solution;

    SSPRK2Integrator<double, FiveMSolutionVec, FluidFluxESDGSEMOperator<dim>> ssp_integrator;
    FluidFluxESDGSEMOperator<dim> fluid_flux_operator;
    unsigned int n_nonmesh_unknowns;
};

template <int dim>
void FiveMomentDGSolver<dim>::reinit() {
    discretization->reinit();
    discretization->perform_allocation(solution.mesh_sol);
    solution.nonmesh_sol.reinit(n_nonmesh_unknowns);
    ssp_integrator.reinit(solution, 3);

    for (unsigned int i = 0; i < species.size(); i++) {
        discretization->project_fluid_quantities(
            *species.at(i)->initial_condition, solution.mesh_sol, i);
    }
}

template <int dim>
void FiveMomentDGSolver<dim>::solve(TimestepCallback writeout_callback) {
    auto step = [&](double t, double dt) -> bool {
        ssp_integrator.evolve_one_time_step(fluid_flux_operator, solution, dt, t);
        return true;
    };
    auto recommend_dt = [&]() -> double { 
        return fluid_flux_operator.recommend_dt(
                discretization->get_matrix_free(),
                solution);
    };
    std::vector<TimestepCallback> callbacks = {writeout_callback};
    advance(step, t_end, recommend_dt, callbacks);
}

template <int dim>
LinearAlgebra::distributed::Vector<double>&
FiveMomentDGSolver<dim>::get_solution() {
    return solution.mesh_sol;
}

}  // namespace five_moment
}  // namespace warpii
