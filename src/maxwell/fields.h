#pragma once
#include "../normalization.h"
#include <deal.II/base/parameter_handler.h>
#include "maxwell.h"
#include "bc_map.h"
#include "phmaxwell_func.h"

namespace warpii {

template<int dim>
class PHMaxwellFields {
   public:
    PHMaxwellFields(double phmaxwell_gamma, double phmaxwell_chi,
           PlasmaNormalization plasma_norm,
           PHMaxwellFunc<dim> initial_condition,
           MaxwellBCMap<dim> bc_map
           )
        : phmaxwell_gamma(phmaxwell_gamma),
          phmaxwell_chi(phmaxwell_chi),
          plasma_norm(plasma_norm),
          initial_condition(initial_condition),
          bc_map(bc_map)
    {}

    PHMaxwellConstants phmaxwell_constants() {
        return PHMaxwellConstants(plasma_norm.speed_of_light(), phmaxwell_chi,
                                  phmaxwell_gamma);
    }

    static void declare_parameters(ParameterHandler& prm,
            unsigned int n_boundaries);

    static std::shared_ptr<PHMaxwellFields> create_from_parameters(ParameterHandler &prm, 
            unsigned int n_boundaries,
            PlasmaNormalization plasma_norm);

    const MaxwellBCMap<dim>& get_bc_map() {
        return bc_map;
    }

   private:
    double phmaxwell_gamma;
    double phmaxwell_chi;
    PlasmaNormalization plasma_norm;
    PHMaxwellFunc<dim> initial_condition;
    MaxwellBCMap<dim> bc_map;
};

template <int dim>
void declare_phmaxwell_funcs(ParameterHandler &prm);

}  // namespace warpii
