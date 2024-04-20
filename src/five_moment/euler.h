#pragma once
#include <deal.II/base/tensor.h>

namespace warpii {
namespace five_moment {
using namespace dealii;

/**
 * Compute the velocity from the conserved variables
 */
template <int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE Tensor<1, dim, Number> euler_velocity(
    const Tensor<1, dim + 2, Number> &conserved_variables) {
    const Number inverse_density = Number(1.) / conserved_variables[0];

    Tensor<1, dim, Number> velocity;
    for (unsigned int d = 0; d < dim; d++) {
        velocity[d] = conserved_variables[d + 1] * inverse_density;
    }

    return velocity;
}

/**
 * Compute Euler pressure from conserved variables
 */
template <int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE Number euler_pressure(
    const Tensor<1, dim + 2, Number> &conserved_variables, double gamma) {
    const Number rho = conserved_variables[0];
    Number squared_momentum = Number(0.);
    for (unsigned int d = 0; d < dim; d++) {
        Number p_d = conserved_variables[d + 1];
        squared_momentum += p_d * p_d;
    }
    const Number kinetic_energy = squared_momentum / (2. * rho);
    const Number total_energy = conserved_variables[dim + 1];
    return (gamma - 1) * (total_energy - kinetic_energy);
}

template <int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE Tensor<1, dim + 2, dealii::Tensor<1, dim, Number>>
euler_flux(const Tensor<1, dim + 2, Number> &q, double gamma) {
    const Tensor<1, dim, Number> velocity = euler_velocity<dim>(q);
    const Number pressure = euler_pressure<dim>(q, gamma);

    Tensor<1, dim + 2, Tensor<1, dim, Number>> flux;
    for (unsigned int d = 0; d < dim; d++) {
        flux[0][d] = q[d + 1];
        for (unsigned int e = 0; e < dim; e++) {
            flux[e + 1][d] = q[e + 1] * velocity[d];
        }
        // Diagonal pressure tensor.
        flux[d + 1][d] += pressure;
        flux[dim + 1][d] = velocity[d] * (q[dim + 1] + pressure);
    }
    return flux;
}

/**
 * Lax-Friedrichs flux
 */
template <int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE Tensor<1, dim + 2, Number> euler_numerical_flux(
    const Tensor<1, dim + 2, Number> &q_in,
    const Tensor<1, dim + 2, Number> &q_out,
    const Tensor<1, dim, Number> &outward_normal, double gamma) {
    const auto u_in = euler_velocity<dim>(q_in);
    const auto u_out = euler_velocity<dim>(q_out);

    const auto pressure_in = euler_pressure<dim>(q_in, gamma);
    const auto pressure_out = euler_pressure<dim>(q_out, gamma);

    const auto flux_in = euler_flux<dim>(q_in, gamma);
    const auto flux_out = euler_flux<dim>(q_out, gamma);

    const auto lambda =
        0.5 * std::sqrt(std::max(
                  u_out.norm_square() + gamma * pressure_out / q_out[0],
                  u_in.norm_square() + gamma * pressure_in / q_in[0]));

    return 0.5 * (flux_in * outward_normal + flux_out * outward_normal) +
           0.5 * lambda * (q_in - q_out);
}

}  // namespace five_moment
}  // namespace warpii
