#pragma once
#include <deal.II/base/exceptions.h>
#include <deal.II/base/parameter_handler.h>

using namespace dealii;

namespace warpii {

/**
 * Implements the flexible plasma normalization described in Coughlin's thesis
 * (2024).
 *
 * The constructor uses the normalization triplet (omega_c_tau, omega_p_tau,
 * n0).
 */
class PlasmaNormalization {
   public:
    PlasmaNormalization(double omega_c_tau, double omega_p_tau, double n0)
        : omega_c_tau(omega_c_tau), omega_p_tau(omega_p_tau), n0(n0) {
        AssertThrow(
            omega_c_tau > 0.0,
            ExcMessage("Nondimensional cyclotron frequency must be positive."));
        AssertThrow(
            omega_p_tau > 0.0,
            ExcMessage("Nondimensional plasma frequency must be positive."));
        AssertThrow(n0 > 0.0,
                    ExcMessage("Reference number density must be positive."));
    }

    static void declare_parameters(ParameterHandler& prm);
    static PlasmaNormalization create_from_parameters(ParameterHandler& prm);

    /**
     * The nondimensional cyclotron frequency.
     *
     * Defaults to 1.0.
     */
    double omega_c_tau;

    /**
     * The nondimensional plasma frequency.
     *
     * Defaults to 1.0.
     */
    double omega_p_tau;

    /**
     * The reference number density in units of m^-3.
     *
     * Defaults to 1e20.
     */
    double n0;

    /**
     * The nondimensional speed of light, expressed in units of the reference velocity `v0`.
     */
    double speed_of_light() {
        return omega_p_tau / omega_c_tau;
    }
};

}  // namespace warpii
