#pragma once

#include <deal.II/base/time_stepping.h>
#include <deal.II/lac/la_parallel_vector.h>

#include "defs.h"

using namespace dealii;

enum LowStorageRungeKuttaScheme {
    stage_3_order_3, /* Kennedy, Carpenter, Lewis, 2000 */
    stage_5_order_4, /* Kennedy, Carpenter, Lewis, 2000 */
    stage_7_order_4, /* Tselios, Simos, 2007 */
    stage_9_order_5, /* Kennedy, Carpenter, Lewis, 2000 */
};

class LowStorageRungeKuttaIntegrator {
   public:
    LowStorageRungeKuttaIntegrator(const LowStorageRungeKuttaScheme scheme) {
        TimeStepping::runge_kutta_method lsrk;
        switch (scheme) {
            case stage_3_order_3: {
                lsrk = TimeStepping::LOW_STORAGE_RK_STAGE3_ORDER3;
                break;
            }

            case stage_5_order_4: {
                lsrk = TimeStepping::LOW_STORAGE_RK_STAGE5_ORDER4;
                break;
            }

            case stage_7_order_4: {
                lsrk = TimeStepping::LOW_STORAGE_RK_STAGE7_ORDER4;
                break;
            }

            case stage_9_order_5: {
                lsrk = TimeStepping::LOW_STORAGE_RK_STAGE9_ORDER5;
                break;
            }

            default:
                AssertThrow(false, ExcNotImplemented());
        }
        TimeStepping::LowStorageRungeKutta<
            LinearAlgebra::distributed::Vector<real>>
            rk_integrator(lsrk);
        rk_integrator.get_coefficients(ai, bi, ci);
    }

    unsigned int n_stages() const { return bi.size(); }

    template <typename VectorType, typename Operator>
    void perform_time_step(const Operator &pde_operator,
                           const double current_time, const double time_step,
                           VectorType &solution, VectorType &vec_ri,
                           VectorType &vec_ki) const {
        AssertDimension(ai.size() + 1, bi.size());

        pde_operator.perform_stage(current_time, bi[0] * time_step,
                                   ai[0] * time_step, solution, vec_ri,
                                   solution, vec_ri);

        for (unsigned int stage = 1; stage < bi.size(); ++stage) {
            const double c_i = ci[stage];
            pde_operator.perform_stage(
                current_time + c_i * time_step, bi[stage] * time_step,
                (stage == bi.size() - 1 ? 0 : ai[stage] * time_step), vec_ri,
                vec_ki, solution, vec_ri);
        }
    }

   private:
    std::vector<double> bi;
    std::vector<double> ai;
    std::vector<double> ci;
};

