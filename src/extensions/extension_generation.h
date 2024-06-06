#pragma once

namespace warpii {

/**
 * Generate an extension in the current directory:
 * - CMakeLists.txt
 * - Makefile
 */
void generate_extension();

void generate_extension_cmakelists();

void generate_extension_makefile();

}
