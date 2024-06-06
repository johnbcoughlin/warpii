#pragma once
#include <string>

namespace warpii {

struct WarpiiOpts {
    /**
     * Whether the setup() method should load input from a file.
     * Defaults to false for convenience for testing.
     *
     * When WarpiiOpts are parsed from a CLI, this is set to true.
     */
    bool input_is_from_file;

    /**
     * The source to read the input file from.
     * Can be a filename, or the string "-", indicating standard input.
     */
    std::string input;

    /**
     * Whether the help flag has been set.
     * If this is set, all other options may have been ignored during parsing.
     */
    bool help;

    /**
     * Whether floating point exceptions (FPEs) are enabled.
     */
    bool fpe;

    /**
     * Whether to stop after the setup phase
     */
    bool setup_only;

    WarpiiOpts() : 
        input_is_from_file(false), input(""), 
        help(false), fpe(false), setup_only(false) {}
};

WarpiiOpts parse_opts(int argc, char **argv);

}  // namespace warpii
