#pragma once
#include <deal.II/base/parameter_handler.h>
#include <iostream>

#include "app.h"
#include "opts.h"
#include "extensions/extension.h"

namespace warpii {
using namespace dealii;

class Warpii {
   public:

    Warpii(): 
        opts(WarpiiOpts()),
        extension(nullptr),
        setup_complete(false), 
        run_complete(false)
    {}

    Warpii(WarpiiOpts opts): 
        opts(opts), 
        extension(nullptr),
        setup_complete(false), 
        run_complete(false) 
    {}

    Warpii(WarpiiOpts opts, std::shared_ptr<Extension> extension): 
        opts(opts), 
        extension(extension),
        setup_complete(false), 
        run_complete(false) 
    {}

    static Warpii create_from_cli(int argc, char** argv);

    static Warpii create_from_cli(
            int argc, char** argv, 
            std::shared_ptr<Extension> extension);

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
    std::shared_ptr<Extension> extension;
    bool setup_complete;
    bool run_complete;
};

std::string format_workdir(
        const ParameterHandler& prm,
        const WarpiiOpts& opts);

}  // namespace warpii
