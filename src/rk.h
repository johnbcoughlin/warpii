#pragma once

#include <deal.II/base/time_stepping.h>
#include <deal.II/lac/la_parallel_vector.h>

#include "defs.h"
#include "dof_utils.h"

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
    void perform_time_step(const Operator& pde_operator,
                           const double current_time, const double time_step,
                           VectorType& solution, VectorType& vec_ri,
                           VectorType& vec_ki) const {
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

template <typename SolutionVec>
class ForwardEulerOperator {
    public:
        ~ForwardEulerOperator() = default;

        /**
         * Compute the single step
         *
         * ```
         * dst = b*dst + a*u + c*dt*f(u)),
         * ```
         *
         * where f is the RHS function provided by this operator.
         *
         * If `alpha` and `beta` take their default values, this reduces
         * to the simple Forward Euler step
         *
         * ```
         * dst = u + dt * f(u)
         * ```
         */
        virtual void perform_forward_euler_step(
                SolutionVec &dst,
                const SolutionVec &u,
                std::vector<SolutionVec> &sol_registers,
                const double dt,
                const double t,
                const double b = 0.0,
                const double a = 1.0,
                const double c = 1.0) = 0;
};

template <typename Number, typename SolutionVec, typename Operator>
class SSPRK2Integrator {
   public:
    SSPRK2Integrator() {}

    void evolve_one_time_step(Operator& forward_euler_operator,
                              // Destination
                              SolutionVec& solution,
                              const double dt,
                              const double t);

    void reinit(const SolutionVec& sol, int sol_register_count);

   private:
    SolutionVec f_1;
    std::vector<SolutionVec> sol_registers;
};

template <typename Number, typename SolutionVec, typename Operator>
void SSPRK2Integrator<Number, SolutionVec, Operator>::evolve_one_time_step(
    Operator& forward_euler_operator,
    SolutionVec& solution,
    const double dt, const double t) {
    forward_euler_operator.perform_forward_euler_step(
        f_1, solution, sol_registers, dt, t);
    forward_euler_operator.perform_forward_euler_step(
        solution, f_1, sol_registers, dt, t + dt, 0.5, 0.5, 0.5);
}

template <typename Number, typename SolutionVec, typename Operator>
void SSPRK2Integrator<Number, SolutionVec, Operator>::reinit(
    const SolutionVec& sol,
    int sol_register_count) {
    f_1.reinit(sol);
    for (int i = 0; i < sol_register_count; i++) {
        sol_registers.emplace_back();
        sol_registers[i].reinit(sol);
    }
}

