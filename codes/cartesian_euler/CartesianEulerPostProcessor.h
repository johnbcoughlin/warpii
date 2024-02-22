#pragma once

#include <cmath>
#include <deal.II/numerics/data_out.h>

#include "cartesian_euler.h"

namespace CartesianEuler {
using namespace dealii;

template<int dim>
class CartesianEulerPostprocessor : public DataPostprocessor<dim> {
   public:
    CartesianEulerPostprocessor(double gamma) : gamma(gamma) {}

    virtual void evaluate_vector_field(
        const DataPostprocessorInputs::Vector<dim> &inputs,
        std::vector<Vector<double>> &computed_quantities) const override;

    virtual std::vector<std::string> get_names() const override;

    virtual std::vector<
        DataComponentInterpretation::DataComponentInterpretation>
    get_data_component_interpretation() const override;

    virtual UpdateFlags get_needed_update_flags() const override;

   private:
    double gamma;
};

template<int dim>
void CartesianEulerPostprocessor<dim>::evaluate_vector_field(
    const DataPostprocessorInputs::Vector<dim> &inputs,
    std::vector<Vector<double>> &computed_quantities) const {
    const unsigned int n_evaluation_points = inputs.solution_values.size();
    const auto points = inputs.evaluation_points;

    Assert(computed_quantities.size() == n_evaluation_points,
           ExcInternalError());
    Assert(inputs.solution_values[0].size() == dim + 2, ExcInternalError());
    Assert(computed_quantities[0].size() == dim + 2, ExcInternalError());

    for (unsigned int p = 0; p < n_evaluation_points; ++p) {
        Tensor<1, dim + 2, double> solution;
        for (unsigned int d = 0; d < dim + 2; ++d) {
            solution[d] = inputs.solution_values[p](d);
        }
        auto density = solution[0];
        const Tensor<1, dim> velocity =
            euler_velocity<dim, double>(solution);
        double pressure = euler_pressure<dim, double>(solution, gamma);

        for (unsigned int d = 0; d < dim; ++d) {
            computed_quantities[p](d) = velocity[d];
        }
        computed_quantities[p](dim) = pressure;
        computed_quantities[p](dim+1) = std::sqrt(gamma * pressure / density);
    }
}

template <int dim>
std::vector<std::string> CartesianEulerPostprocessor<dim>::get_names() const {
    std::vector<std::string> names;
    for (unsigned int d = 0; d < dim; ++d) {
        names.emplace_back("velocity");
    }
    names.emplace_back("pressure");
    names.emplace_back("speed_of_sound");

    return names;
}

template <int dim>
UpdateFlags CartesianEulerPostprocessor<dim>::get_needed_update_flags() const {
    return update_values | dealii::update_quadrature_points;
}

template <int dim>
std::vector<DataComponentInterpretation::DataComponentInterpretation>
CartesianEulerPostprocessor<dim>::get_data_component_interpretation() const {
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
        interpretation;
    // velocity
    for (unsigned int d = 0; d < dim; ++d)
        interpretation.push_back(
            DataComponentInterpretation::component_is_part_of_vector);

    // pressure
    interpretation.push_back(DataComponentInterpretation::component_is_scalar);

    // speed of sound
    interpretation.push_back(DataComponentInterpretation::component_is_scalar);

    return interpretation;
}
}  // namespace CylindricalEuler
