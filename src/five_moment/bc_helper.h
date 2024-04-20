#pragma once
#include <deal.II/base/function.h>
#include <deal.II/base/types.h>

using namespace dealii;

template <int dim>
class EulerBCMap {
    public:
    void set_inflow_boundary(const types::boundary_id boundary_id,
                             std::unique_ptr<Function<dim>> inflow_function);
    void set_subsonic_outflow_boundary(
        const types::boundary_id boundary_id,
        std::unique_ptr<Function<dim>> outflow_energy);
    void set_supersonic_outflow_boundary(const types::boundary_id boundary_id);
    void set_axial_boundary(const types::boundary_id boundary_id);
    void set_wall_boundary(const types::boundary_id boundary_id);

    bool is_inflow(const types::boundary_id boundary_id) const;
    bool is_subsonic_outflow(const types::boundary_id boundary_id) const;
    bool is_supersonic_outflow(const types::boundary_id boundary_id) const;
    bool is_axial(const types::boundary_id boundary_id) const;
    bool is_wall(const types::boundary_id boundary_id) const;

    std::shared_ptr<Function<dim>> get_inflow(const types::boundary_id boundary_id) const {
        return _inflow_boundaries.find(boundary_id)->second;
    }
    std::shared_ptr<Function<dim>> get_subsonic_outflow_energy(const types::boundary_id boundary_id) const {
        return _subsonic_outflow_boundaries.find(boundary_id)->second;
    }
    const std::map<types::boundary_id, std::shared_ptr<Function<dim>>>& inflow_boundaries() const {
        return _inflow_boundaries;
    }
    const std::map<types::boundary_id, std::shared_ptr<Function<dim>>>& subsonic_outflow_boundaries() const {
        return _subsonic_outflow_boundaries;
    }

    private:
    std::map<types::boundary_id, std::shared_ptr<Function<dim>>>
        _inflow_boundaries;
    std::map<types::boundary_id, std::shared_ptr<Function<dim>>>
        _subsonic_outflow_boundaries;
    std::set<types::boundary_id> supersonic_outflow_boundaries;
    std::set<types::boundary_id> axial_boundaries;
    std::set<types::boundary_id> wall_boundaries;

    // Helper set of the boundary ids that are already spoken for.
    std::set<types::boundary_id> set_boundary_ids;
};

template <int dim>
void EulerBCMap<dim>::set_inflow_boundary(
    const types::boundary_id boundary_id,
    std::unique_ptr<Function<dim>> inflow_function) {
    AssertThrow(set_boundary_ids.find(boundary_id) == set_boundary_ids.end(),
                ExcMessage("You already set the boundary with id " +
                           std::to_string(static_cast<int>(boundary_id)) +
                           " to another type of boundary before now setting " +
                           "it as an axis boundary"));
    AssertThrow(inflow_function->n_components == dim + 2,
                ExcMessage("Expected function with dim+2 components"));

    _inflow_boundaries[boundary_id] = std::move(inflow_function);
}

template <int dim>
void EulerBCMap<dim>::set_subsonic_outflow_boundary(
    const types::boundary_id boundary_id,
    std::unique_ptr<Function<dim>> outflow_function) {
    AssertThrow(set_boundary_ids.find(boundary_id) == set_boundary_ids.end(),
                ExcMessage("You already set the boundary with id " +
                           std::to_string(static_cast<int>(boundary_id)) +
                           " to another type of boundary before now setting " +
                           "it as an axis boundary"));
    AssertThrow(outflow_function->n_components == dim + 2,
                ExcMessage("Expected function with dim+2 components"));

    set_boundary_ids.insert(boundary_id);
    _subsonic_outflow_boundaries[boundary_id] = std::move(outflow_function);
}

template <int dim>
void EulerBCMap<dim>::set_supersonic_outflow_boundary(
    const types::boundary_id boundary_id) {
    AssertThrow(set_boundary_ids.find(boundary_id) == set_boundary_ids.end(),
                ExcMessage("You already set the boundary with id " +
                           std::to_string(static_cast<int>(boundary_id)) +
                           " to another type of boundary before now setting " +
                           "it as supersonic outflow"));

    set_boundary_ids.insert(boundary_id);
    supersonic_outflow_boundaries.insert(boundary_id);
}

template <int dim>
void EulerBCMap<dim>::set_axial_boundary(
    const types::boundary_id boundary_id) {
    AssertThrow(set_boundary_ids.find(boundary_id) == set_boundary_ids.end(),
                ExcMessage("You already set the boundary with id " +
                           std::to_string(static_cast<int>(boundary_id)) +
                           " to another type of boundary before now setting " +
                           "it as an axis boundary"));

    set_boundary_ids.insert(boundary_id);
    axial_boundaries.insert(boundary_id);
}

template <int dim>
void EulerBCMap<dim>::set_wall_boundary(
    const types::boundary_id boundary_id) {
    AssertThrow(set_boundary_ids.find(boundary_id) == set_boundary_ids.end(),
                ExcMessage("You already set the boundary with id " +
                           std::to_string(static_cast<int>(boundary_id)) +
                           " to another type of boundary before now setting " +
                           "it as a wall boundary"));

    set_boundary_ids.insert(boundary_id);
    wall_boundaries.insert(boundary_id);
}

template <int dim>
bool EulerBCMap<dim>::is_inflow(const types::boundary_id boundary_id) const {
    return _inflow_boundaries.find(boundary_id) != _inflow_boundaries.end();
}
template <int dim>
bool EulerBCMap<dim>::is_subsonic_outflow(const types::boundary_id boundary_id) const {
    return _subsonic_outflow_boundaries.find(boundary_id) != _subsonic_outflow_boundaries.end();
}
template <int dim>
bool EulerBCMap<dim>::is_supersonic_outflow(const types::boundary_id boundary_id) const {
    return supersonic_outflow_boundaries.find(boundary_id) != supersonic_outflow_boundaries.end();
}
template <int dim>
bool EulerBCMap<dim>::is_axial(const types::boundary_id boundary_id) const {
    return axial_boundaries.find(boundary_id) != axial_boundaries.end();
}
template <int dim>
bool EulerBCMap<dim>::is_wall(const types::boundary_id boundary_id) const {
    return wall_boundaries.find(boundary_id) != wall_boundaries.end();
}
