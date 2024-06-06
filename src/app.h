#pragma once

#include "opts.h"
#include "extensions/extension.h"

namespace warpii {
class AbstractApp {
    public:
        virtual ~AbstractApp() = default;

        /**
         * Set up the application:
         * - Allocate solution vectors
         * - Construct meshes
         * - For time-dependent problems, evaluate and project initial conditions
         * - Write out output files that define the problem:
         *   - Mesh visualizations
         *   - Canonical input files
         */
        virtual void setup(WarpiiOpts opts) = 0;

        /**
         * Run the simulation / solve the problem.
         */
        virtual void run(WarpiiOpts opts) = 0;
};
}
