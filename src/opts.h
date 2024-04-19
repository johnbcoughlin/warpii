#pragma once
#include <string>

namespace warpii {

struct WarpiiOpts {
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

    WarpiiOpts() : input(""), help(false) {}
};

WarpiiOpts parse_opts(int argc, char **argv);

}  // namespace warpii
