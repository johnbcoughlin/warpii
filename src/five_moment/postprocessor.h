#pragma once

#include "euler.h"
#include <deal.II/numerics/data_out.h>

#include <cmath>

namespace warpii {
namespace five_moment {
using namespace dealii;

template <int dim>
class FiveMomentPostprocessor : public DataPostprocessor<dim> {
   public:
    FiveMomentPostprocessor(double gamma) : gamma(gamma) {}

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

template <int dim>
void FiveMomentPostprocessor<dim>::evaluate_vector_field(
    const DataPostprocessorInputs::Vector<dim> &inputs,
    std::vector<Vector<double>> &computed_quantities) const {
    const unsigned int n_evaluation_points = inputs.solution_values.size();
    const auto points = inputs.evaluation_points;

    Assert(computed_quantities.size() == n_evaluation_points,
           ExcInternalError());
    Assert(inputs.solution_values[0].size() == 5, ExcInternalError());
    Assert(computed_quantities[0].size() == 6, ExcInternalError());

    for (unsigned int p = 0; p < n_evaluation_points; ++p) {
        Tensor<1, 5, double> solution;
        for (unsigned int comp = 0; comp < 5; ++comp) {
            solution[comp] = inputs.solution_values[p](comp);
        }
        auto density = solution[0];
        const Tensor<1, 3> velocity = euler_velocity<3, double>(solution);
        double pressure = euler_pressure<dim, double>(solution, gamma);

        for (unsigned int d = 0; d < 3; ++d) {
            computed_quantities[p](d) = velocity[d];
        }
        computed_quantities[p](3) = pressure;
        computed_quantities[p](4) =
            std::log(pressure) - gamma * std::log(density);
        computed_quantities[p](5) = std::sqrt(gamma * pressure / density);
    }
}

template <int dim>
std::vector<std::string> FiveMomentPostprocessor<dim>::get_names() const {
    std::vector<std::string> names;
    names.emplace_back("x_velocity");
    names.emplace_back("y_velocity");
    names.emplace_back("z_velocity");
    names.emplace_back("pressure");
    names.emplace_back("specific_entropy");
    names.emplace_back("speed_of_sound");

    return names;
}

template <int dim>
UpdateFlags FiveMomentPostprocessor<dim>::get_needed_update_flags() const {
    return update_values | dealii::update_quadrature_points;
}

template <int dim>
std::vector<DataComponentInterpretation::DataComponentInterpretation>
FiveMomentPostprocessor<dim>::get_data_component_interpretation() const {
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
        interpretation;
    // velocity
    for (unsigned int d = 0; d < 3; ++d)
        interpretation.push_back(
            DataComponentInterpretation::component_is_scalar);

    // pressure
    interpretation.push_back(DataComponentInterpretation::component_is_scalar);

    // entropy
    interpretation.push_back(DataComponentInterpretation::component_is_scalar);

    // speed of sound
    interpretation.push_back(DataComponentInterpretation::component_is_scalar);

    return interpretation;
}
}  // namespace five_moment
}  // namespace warpii
