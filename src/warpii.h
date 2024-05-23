#pragma once
#include <deal.II/base/parameter_handler.h>
#include <iostream>

#include "app.h"
#include "opts.h"

namespace warpii {
using namespace dealii;

class Warpii {
   public:
    Warpii(): setup_complete(false), run_complete(false) {}
    Warpii(WarpiiOpts opts) : opts(opts), setup_complete(false), run_complete(false) {}

    static Warpii create_from_cli(int argc, char** argv);

    /**
     * Loads the `input` string from the file specified by `opts`.
     */
    void load_input_from_file();

    static void print_help(bool to_err = false);

    /**
     * Perform the setup phase of the problem.
     * Calls AbstractApp::setup()
     */
    void setup();

    /**
     * Run the simulation / solve the problem.
     * If setup() has not been called yet, this method calls it.
     */
    void run();

    template <typename AppImpl>
    AppImpl& get_app() {
        return *(dynamic_cast<AppImpl*>(app.get()));
    }

    WarpiiOpts opts;
    std::string input;

   private:
    ParameterHandler prm;
    std::unique_ptr<AbstractApp> app;
    bool setup_complete;
    bool run_complete;
};

}  // namespace warpii
