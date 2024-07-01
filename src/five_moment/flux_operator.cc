#include "flux_operator.h"

#include "solution_vec.h"

namespace warpii {
namespace five_moment {

template <int dim>
void FiveMomentFluxOperator<dim>::perform_forward_euler_step(
    FiveMSolutionVec &dst, const FiveMSolutionVec &u,
    std::vector<FiveMSolutionVec> &sol_registers, const double dt,
    const double t, const double b, const double a, const double c) {
    // dst1 = b*dst + a*u + c*dt*f1(u)
    fluid_flux.perform_forward_euler_step(dst, u, sol_registers, dt, t, b, a,
                                          c);

    if (fields_enabled) {
        // dst = 1*dst1 + 0*u + c*dt*f2(u)
        //     = b*dst + a*u + c*dt*(f1(u) + f2(u))
        maxwell_flux.perform_forward_euler_step(
                dst, u, sol_registers, dt, t, 1.0, 0.0, c);
    }
}

template class FiveMomentFluxOperator<1>;
template class FiveMomentFluxOperator<2>;

}  // namespace five_moment
}  // namespace warpii
