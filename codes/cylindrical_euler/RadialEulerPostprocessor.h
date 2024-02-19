#pragma once

#include <cmath>
#include <deal.II/numerics/data_out.h>

#include "radial_euler.h"

namespace CylindricalEuler {
using namespace dealii;

template<int dim>
class RadialEulerPostprocessor : public DataPostprocessor<dim> {
   public:
    RadialEulerPostprocessor(double gamma) : gamma(gamma) {}

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
void RadialEulerPostprocessor<dim>::evaluate_vector_field(
    const DataPostprocessorInputs::Vector<dim> &inputs,
    std::vector<Vector<double>> &computed_quantities) const {
    const unsigned int n_evaluation_points = inputs.solution_values.size();
    const auto points = inputs.evaluation_points;

    Assert(computed_quantities.size() == n_evaluation_points,
           ExcInternalError());
    Assert(inputs.solution_values[0].size() == dim + 2, ExcInternalError());
    Assert(computed_quantities[0].size() == 2*dim + 4, ExcInternalError());

    for (unsigned int p = 0; p < n_evaluation_points; ++p) {
        Point<dim> pt = points[p];
        double r = pt[0];

        Tensor<1, dim + 2, double> solution;
        for (unsigned int d = 0; d < dim + 2; ++d) {
            solution[d] = inputs.solution_values[p](d);
        }

        double density, energy;
        Tensor<1, dim> momentum;
        if (std::abs(r) > 1e-15) {
            density = solution[0] / r;
            for (unsigned int d = 0; d < dim; ++d) {
                momentum[d] = solution[1+d] / r;
            }
            energy = solution[dim+1] / r;
        } else {
            density = std::nan(0);
            for (unsigned int d = 0; d < dim; ++d) {
                momentum[d] = std::nan(0);
            }
            energy = std::nan(0);
        }

        const Tensor<1, dim> velocity =
            radial_euler_velocity<dim, double>(solution);
        double pressure;
        if (std::abs(r) > 1e-15) {
            pressure = radial_euler_pressure<dim, double>(solution, gamma) / r;
        } else {
            pressure = std::nan(0);
        }

        computed_quantities[p](0) = density;
        for (unsigned int d = 0; d < dim; ++d) {
            computed_quantities[p](1+d) = momentum[d];
        }
        computed_quantities[p](dim+1) = energy;
        for (unsigned int d = 0; d < dim; ++d) {
            computed_quantities[p](dim+2 + d) = velocity[d];
        }
        computed_quantities[p](2*dim+2) = pressure;
        computed_quantities[p](2*dim+3) = std::sqrt(gamma * pressure / density);
    }
}

template <int dim>
std::vector<std::string> RadialEulerPostprocessor<dim>::get_names() const {
    std::vector<std::string> names;
    names.emplace_back("density");
    for (unsigned int d = 0; d < dim; ++d) {
        names.emplace_back("momentum");
    }
    names.emplace_back("energy");
    for (unsigned int d = 0; d < dim; ++d) {
        names.emplace_back("velocity");
    }
    names.emplace_back("pressure");
    names.emplace_back("speed_of_sound");

    return names;
}

template <int dim>
UpdateFlags RadialEulerPostprocessor<dim>::get_needed_update_flags() const {
    return update_values | dealii::update_quadrature_points;
}

template <int dim>
std::vector<DataComponentInterpretation::DataComponentInterpretation>
RadialEulerPostprocessor<dim>::get_data_component_interpretation() const {
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
        interpretation;

    // density
    interpretation.push_back(DataComponentInterpretation::component_is_scalar);
    // momentum
    for (unsigned int d = 0; d < dim; ++d)
        interpretation.push_back(
            DataComponentInterpretation::component_is_part_of_vector);
    // energy
    interpretation.push_back(DataComponentInterpretation::component_is_scalar);

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
