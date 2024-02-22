#pragma once

#include <deal.II/base/config.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include "tensor_utils.h"

namespace CartesianEuler {
using namespace dealii;

// Returns u
template <int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE Tensor<1, dim, Number> euler_velocity(
    const Tensor<1, dim + 2, Number> &conserved_variables) {
    const Number rho_inverse = Number(1.) / conserved_variables[0];
    Tensor<1, dim, Number> velocity;

    for (unsigned int d = 0; d < dim; d++) {
        velocity[d] = conserved_variables[1 + d] * rho_inverse;
    }

    return velocity;
}

// Returns pressure
template <int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE Number euler_pressure(
    const Tensor<1, dim + 2, Number> &conserved_variables, double gamma) {
    const Tensor<1, dim, Number> velocity =
        euler_velocity<dim, Number>(conserved_variables);

    Number rho_u_dot_u = conserved_variables[1] * velocity[0];
    for (unsigned int d = 1; d < dim; d++) {
        rho_u_dot_u += conserved_variables[d + 1] * velocity[d];
    }

    return (gamma = 1.) * (conserved_variables[dim + 1] - 0.5 * rho_u_dot_u);
}

template <int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE Tensor<1, dim + 2, Tensor<1, dim, Number>>
euler_flux(const Tensor<1, dim + 2, Number> &conserved_variables,
           double gamma) {
    const Tensor<1, dim, Number> velocity =
        euler_velocity<dim>(conserved_variables);
    const Number pressure =
        euler_pressure<dim, Number>(conserved_variables, gamma);

    Tensor<1, dim + 2, Tensor<1, dim, Number>> flux;
    for (unsigned int d = 0; d < dim; d++) {
        flux[0][d] = conserved_variables[d + 1];
        for (unsigned int e = 0; e < dim; e++) {
            flux[e + 1][d] = conserved_variables[e + 1] * velocity[d];
        }
        flux[d + 1][d] += pressure;
        flux[dim + 1][d] =
            velocity[d] * (conserved_variables[dim + 1] + pressure);
    }

    return flux;
}

template <int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE Tensor<1, dim + 2, Number> euler_numerical_flux(
    const Tensor<1, dim + 2, Number> &u_m,
    const Tensor<1, dim + 2, Number> &u_p, 
    const Tensor<1, dim, Number> &normal,
    double gamma) {
    const auto velocity_m = euler_velocity<dim>(u_m);
    const auto velocity_p = euler_velocity<dim>(u_p);

    const auto pressure_m = euler_pressure<dim>(u_m, gamma);
    const auto pressure_p = euler_pressure<dim>(u_p, gamma);

    const auto flux_m = euler_flux<dim>(u_m, gamma);
    const auto flux_p = euler_flux<dim>(u_p, gamma);

    const auto lambda =
        0.5 *
        std::sqrt(std::max(
            velocity_p.norm_square() + gamma * pressure_p * (1. / u_p[0]),
            velocity_m.norm_square() + gamma * pressure_m * (1. / u_m[0])));

    return 0.5 * (flux_m * normal + flux_p * normal) +
           0.5 * lambda * (u_m - u_p);
}
}  // namespace CartesianEuler
