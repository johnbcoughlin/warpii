#pragma once
#include <deal.II/base/parameter_handler.h>
#include <iostream>

#include "app.h"
#include "opts.h"

namespace warpii {
using namespace dealii;

class Warpii {
   public:
    Warpii() {}
    Warpii(WarpiiOpts opts) : opts(opts) {}

    static Warpii create_from_cli(int argc, char** argv);

    /**
     * Loads the `input` string from the file specified by `opts`.
     */
    void load_input_from_file();

    static void print_help(bool to_err = false);

    void run();

    WarpiiOpts opts;
    std::string input;

   private:
    ParameterHandler prm;
};

}  // namespace warpii
