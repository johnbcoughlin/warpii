#include <deal.II/base/parameter_handler.h>
#include "normalization.h"

using namespace dealii;

namespace warpii {
    void PlasmaNormalization::declare_parameters(
            ParameterHandler& prm) {
        prm.enter_subsection("Normalization");
        prm.declare_entry("omega_c_tau", "1.0",
                Patterns::Double(std::numeric_limits<double>::epsilon()),
                "The nondimensional proton cyclotron (Larmor) frequency.");

        prm.declare_entry("omega_p_tau", "1.0",
                Patterns::Double(std::numeric_limits<double>::epsilon()),
                "The nondimensional proton plasma (Langmuir) frequency.");

        prm.declare_entry("n0", "1.0e20",
                Patterns::Double(std::numeric_limits<double>::epsilon()),
                "The reference number density in units of m^-3.");
        prm.leave_subsection();
    }

    PlasmaNormalization PlasmaNormalization::create_from_parameters(ParameterHandler &prm) {
        prm.enter_subsection("Normalization");
        double omega_c_tau = prm.get_double("omega_c_tau");
        double omega_p_tau = prm.get_double("omega_p_tau");
        double n0 = prm.get_double("n0");
        prm.leave_subsection();
        return PlasmaNormalization(omega_c_tau, omega_p_tau, n0);
    }
}
