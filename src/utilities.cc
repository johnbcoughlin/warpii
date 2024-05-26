#include "utilities.h"

#include <deal.II/base/numbers.h>
#include <iomanip>
#include <string>
#include <vector>
#include <iostream>

void show_for_debug(bool val)
{
    std::cout << val << std::endl;
}

void show_for_debug(int val)
{
    std::cout << val << std::endl;
}

void show_for_debug(unsigned int val)
{
    std::cout << val << std::endl;
}

void show_for_debug(double val)
{
    std::cout << std::setprecision(15) << val << std::endl;
}

void show_for_debug(size_t val)
{
    std::cout << val << std::endl;
}

void show_for_debug(const std::vector<double>& val)
{
    std::cout << "[";
    if (val.size() >= 1)
    {
        std::cout << val[0];
        for (size_t i = 1; i < val.size(); i++)
        {
            std::cout << ", " << val[i];
        }
    }
    std::cout << "]" << std::endl;
}

void show_for_debug(const dealii::VectorizedArray<double>& val)
{
    std::cout << "[";
    if (val.size() >= 1)
    {
        std::cout << val[0];
        for (size_t i = 1; i < val.size(); i++)
        {
            std::cout << ", " << val[i];
        }
    }
    std::cout << "]" << std::endl;
}
