#pragma once
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/patterns.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <memory>

#include "../app.h"
#include "../extensions/extension.h"
#include "../wrapper.h"
#include "extension.h"

using namespace dealii;

namespace warpii {
namespace cylindrical_five_moment {

class CylindricalFiveMomentWrapper : public ApplicationWrapper {
    public:
        void declare_parameters(ParameterHandler& prm) override;

        std::unique_ptr<AbstractApp> create_app(ParameterHandler &prm,
                std::string input, std::shared_ptr<warpii::Extension> ext) override;
};

template <int dim>
std::shared_ptr<cylindrical_five_moment::Extension<dim>> unwrap_extension(std::shared_ptr<warpii::Extension> ext) {
    if (!ext) {
        return std::make_shared<cylindrical_five_moment::Extension<dim>>();
    }
    if (auto result = std::dynamic_pointer_cast<cylindrical_five_moment::Extension<dim>>(ext)) {
        return result;
    }
    return std::make_shared<cylindrical_five_moment::Extension<dim>>();
}

}
}  // namespace warpii
