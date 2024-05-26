#pragma once

#include <deal.II/base/vectorization.h>
#include <vector>
#include <cstddef>

template <typename T>
void show_for_debug(T val) {
    std::cout << val << std::endl;
}

void show_for_debug(bool val);

void show_for_debug(int val);

void show_for_debug(unsigned int val);

void show_for_debug(size_t val);

void show_for_debug(double val);

void show_for_debug(const std::vector<double>& val);

void show_for_debug(const dealii::VectorizedArray<double>& val);

#ifdef NDEBUG
#define SHOW(varname) // nothing here
#else
#define SHOW(varname)                                                                    \
    std::cout << #varname << " = ";                                                      \
    show_for_debug(varname);
#endif
