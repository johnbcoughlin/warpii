#pragma once

#include "opts.h"

namespace warpii {
class AbstractApp {
    public:
        virtual ~AbstractApp() = default;

        virtual void run(WarpiiOpts opts) = 0;
};
}
