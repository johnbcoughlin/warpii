#pragma once

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include "tensor_utils.h"

namespace CylindricalEuler {
using namespace dealii;

// Returns u
template <int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE Tensor<1, dim, Number> radial_euler_velocity(
    const Tensor<1, dim + 2, Number> &conserved_variables) {
    const Number r_rho_inverse = Number(1.) / conserved_variables[0];

    Tensor<1, dim, Number> velocity;

    for (unsigned int d = 0; d < dim; ++d) {
        velocity[d] = conserved_variables[1+d] * r_rho_inverse;
    }

    return velocity;
}

// Returns rp
template <int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE Number radial_euler_pressure(
    const Tensor<1, dim + 2, Number> &conserved_variables, double gamma) {
    const Tensor<1, dim, Number> velocity =
        radial_euler_velocity<dim, Number>(conserved_variables);

    Number r_rho_u_dot_u = conserved_variables[1] * velocity[0];
    for (unsigned int d = 1; d < dim; ++d) {
        r_rho_u_dot_u += conserved_variables[1 + d] * velocity[d];
    }

    return (gamma - 1.) * (conserved_variables[dim + 1] - 0.5 * r_rho_u_dot_u);
}

template <int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE Tensor<1, dim + 2, dealii::Tensor<1, dim, Number>>
radial_euler_flux(const Tensor<1, dim + 2, Number> &conserved_variables, double gamma, bool log=false) {
    // Conserved variables are r*rho, r*rho*u, and r*E.
    const Tensor<1, dim, Number> velocity =
        radial_euler_velocity<dim>(conserved_variables);
    const Number r_pressure = radial_euler_pressure<dim, Number>(conserved_variables, gamma);

    if (log) {
        std::cout << "q: " << conserved_variables << std::endl;
        std::cout << "velocity: " << velocity << std::endl;
    }

    Tensor<1, dim + 2, Tensor<1, dim, Number>> flux;
    for (unsigned int d = 0; d < dim; ++d) {
        flux[0][d] = conserved_variables[1 + d];
        for (unsigned int e = 0; e < dim; ++e) {
            flux[e + 1][d] = conserved_variables[e + 1] * velocity[d];
        }
        flux[d + 1][d] += r_pressure;
        flux[dim + 1][d] =
            velocity[d] * (conserved_variables[dim + 1] + r_pressure);
    }

    //std::cout << "Nonzero radial density flux: " << flux[0][0] << std::endl;
    return flux;
}

template <int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE Tensor<1, dim + 2, Number> radial_euler_numerical_flux(
    const Tensor<1, dim + 2, Number> &u_m,
    const Tensor<1, dim + 2, Number> &u_p, 
    const Tensor<1, dim, Number> &normal,
    const Point<dim, Number> &,
    double gamma,
    bool log=false) {
    const auto velocity_m = radial_euler_velocity<dim>(u_m);
    const auto velocity_p = radial_euler_velocity<dim>(u_p);

    const auto r_pressure_m = radial_euler_pressure<dim>(u_m, gamma);
    const auto r_pressure_p = radial_euler_pressure<dim>(u_p, gamma);

    const auto flux_m = radial_euler_flux<dim>(u_m, gamma, log);
    const auto flux_p = radial_euler_flux<dim>(u_p, gamma, log);

    const auto lambda =
        0.5 *
        std::sqrt(std::max(
            velocity_p.norm_square() + gamma * r_pressure_p * (1. / u_p[0]),
            velocity_m.norm_square() + gamma * r_pressure_m * (1. / u_m[0])));

    if (log) {
                std::cout << "interior flux: " << flux_m << std::endl;
                std::cout << "exterior flux: " << flux_p << std::endl;
    }


    return 0.5 * (flux_m * normal + flux_p * normal) +
           0.5 * lambda * (u_m - u_p);
}

}  // namespace CylindricalEuler
