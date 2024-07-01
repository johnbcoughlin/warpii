#include "fluid_flux_es_dgsem_operator.h"
#include "../maxwell/maxwell_flux_dg_operator.h"
#include "solution_vec.h"
#include "../rk.h"
#include "../maxwell/fields.h"

namespace warpii {

namespace five_moment {

template <int dim>
class FiveMomentFluxOperator : public ForwardEulerOperator<FiveMSolutionVec> {
    public:
        FiveMomentFluxOperator(
            std::shared_ptr<NodalDGDiscretization<dim>> discretization,
            double gas_gamma, 
            std::vector<std::shared_ptr<Species<dim>>> species,
            std::shared_ptr<PHMaxwellFields<dim>> fields,
            bool fields_enabled
                ):
            fields_enabled(fields_enabled),
            fluid_flux(discretization, gas_gamma, species),
            maxwell_flux(discretization, 5*species.size(), fields)
        {}


    void perform_forward_euler_step(
        FiveMSolutionVec &dst,
        const FiveMSolutionVec &u,
        std::vector<FiveMSolutionVec> &sol_registers,
        const double dt, const double t, 
        const double b = 0.0,
        const double a = 1.0,
        const double c = 1.0) override;

    private:
    bool fields_enabled;
        FluidFluxESDGSEMOperator<dim> fluid_flux;
        MaxwellFluxDGOperator<dim, FiveMSolutionVec> maxwell_flux;
};

}

}
