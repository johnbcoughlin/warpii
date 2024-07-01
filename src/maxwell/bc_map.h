#pragma once
#include <deal.II/base/function.h>
#include <deal.II/base/types.h>
#include "phmaxwell_func.h"

using namespace dealii;

namespace warpii {

enum MaxwellBCType {
    PERFECT_CONDUCTOR,
    DIRICHLET
};

template <int dim>
class MaxwellBCMap {
    public:
        MaxwellBCMap() {}

        void set_perfect_conductor_boundary(
                const types::boundary_id boundary_id);

        void set_dirichlet_boundary(
                const types::boundary_id boundary_id,
                PHMaxwellFunc<dim> func);

        MaxwellBCType get_bc_type(
                const types::boundary_id boundary_id) const {
            return bcs.at(boundary_id);
        }

        std::shared_ptr<PHMaxwellFunc<dim>> get_dirichlet_func(
                const types::boundary_id boundary_id) const {
            return function_bcs.at(boundary_id);
        }


    private:
        std::map<types::boundary_id, MaxwellBCType> bcs;
        std::map<types::boundary_id, std::shared_ptr<PHMaxwellFunc<dim>>> function_bcs;
};
    
template <int dim>
void MaxwellBCMap<dim>::set_perfect_conductor_boundary(const types::boundary_id boundary_id) {
    AssertThrow(bcs.find(boundary_id) == bcs.end(),
            ExcMessage("Boundary " + std::to_string(static_cast<int>(boundary_id))
                + " was already assigned a boundary condition."));

    bcs.insert(boundary_id, MaxwellBCType::PERFECT_CONDUCTOR);
}

template <int dim>
void MaxwellBCMap<dim>::set_dirichlet_boundary(
        const types::boundary_id boundary_id,
        PHMaxwellFunc<dim> func) {
    AssertThrow(bcs.find(boundary_id) == bcs.end(),
            ExcMessage("Boundary " + std::to_string(static_cast<int>(boundary_id))
                + " was already assigned a boundary condition."));

    bcs.insert(boundary_id, MaxwellBCType::DIRICHLET);
    function_bcs.insert(boundary_id, std::move(func));
}

}
