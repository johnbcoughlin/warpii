#pragma once

#include "../extensions/extension.h"
#include <deal.II/base/parameter_handler.h>
#include <deal.II/grid/tria.h>

namespace warpii {

    namespace five_moment {
        template <int dim>
        class Extension : public virtual warpii::Extension, public virtual GridExtension<dim> {

        public:
            Extension() {}
        };
    }

}
