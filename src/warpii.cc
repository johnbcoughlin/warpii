#include <cstdio>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/patterns.h>
#include <iostream>
#include <sstream>
#include "warpii.h"
#include "grid.h"
#include "five_moment/five_moment.h"
#include "opts.h"
#include "wrapper.h"
#include "fpe.h"

namespace warpii {
using namespace dealii;

char* getCmdOption(char ** begin, char ** end, const std::string & option)
{
    char ** itr = std::find(begin, end, option);
    if (itr != end && ++itr != end)
    {
        return *itr;
    }
    return 0;
}

bool cmdOptionExists(char** begin, char** end, const std::string& option)
{
    return std::find(begin, end, option) != end;
}

WarpiiOpts parse_opts(int argc, char **argv) {
    WarpiiOpts opts = WarpiiOpts();

    std::vector<std::string> all_args;

    // Skip the executable name by starting at 1
    for (int i = 1; i < argc; i++) {
        all_args.push_back(std::string(argv[i]));
    }

    bool set_input = false;
    for (size_t i = 0; i < all_args.size(); i++) {
        auto& arg = all_args.at(i);
        if (arg == "--help" || arg == "-h") {
            opts.help = true;
            break;
        } else if (arg == "--enable-fpe") {
            opts.fpe = true;
        } else {
            opts.input = arg;
            set_input = true;
        }
    }

    if (!set_input && !opts.help) {
        std::cout << "Error: no input source was requested." << std::endl;
        Warpii::print_help(true);
        exit(1);
    }

    return opts;
}

Warpii Warpii::create_from_cli(int argc, char **argv) {
    WarpiiOpts opts = parse_opts(argc, argv);
    return Warpii(opts);
}

// Options are specified in the docopt format: http://docopt.org/
void Warpii::print_help(bool to_err) {
    auto& stream = to_err ? std::cerr : std::cout;
    stream << R"(
WarpII: A collection of plasma codes.

Usage:
  warpii [options] <input_file>
  warpii --help | -h

Options:
  --enable-fpe: Defaults to false.
                If this flag is enabled, floating point exceptions will be enabled.
                This means that any NaNs produced during the simulation will result
                in a full stop of the simulation.
                When disabled, NaNs are silently propagated until, possibly, caught
                down the line by some dealii function.
)";
}

void Warpii::load_input_from_file() {
    std::stringstream ss;
    if (opts.input == "-") {
        ss << std::cin.rdbuf();
    } else {
        std::ifstream file(opts.input);
        if (!file.is_open()) {
            std::cerr << "Could not open requested input file <" << opts.input << "> for reading." << std::endl;
            Warpii::print_help();
            exit(1);
        }
        ss << file.rdbuf();
    }
    this->input = ss.str();
}

void Warpii::run() { 
    if (opts.help) {
        Warpii::print_help();
        exit(0);
    }

    if (opts.fpe) {
        enable_floating_point_exceptions();
    }

    prm.declare_entry("Application", "FiveMoment", Patterns::Selection("FiveMoment"));
    prm.parse_input_from_string(input, "", true);

    std::unique_ptr<ApplicationWrapper> app_wrapper;
    if (prm.get("Application") == "FiveMoment") {
        app_wrapper = std::make_unique<five_moment::FiveMomentWrapper>();
    }
    app_wrapper->declare_parameters(prm);
    app = app_wrapper->create_app(prm, input);
    app->run(opts);
}


}  // namespace warpii
