#pragma once

#include <vector>
#include <cstddef>

void show_for_debug(bool val);

void show_for_debug(int val);

void show_for_debug(size_t val);

void show_for_debug(double val);

void show_for_debug(const std::vector<double>& val);

#ifdef NDEBUG
#define SHOW(varname) // nothing here
#else
#define SHOW(varname)                                                                    \
    std::cout << #varname << " = ";                                                      \
    show_for_debug(varname);
#endif
