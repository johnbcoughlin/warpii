#include <deal.II/base/tensor.h>

using namespace dealii;

// Returns a random real number between 0 and 1
double rand_01(); 

template <int dim>
Tensor<1, dim+2, double> random_euler_state(double gas_gamma) {
    Tensor<1, dim+2, double> state;
    Tensor<1, dim, double> u;
    double rho = (rand_01() + 1e-10) * 100;
    for (unsigned int d = 0; d < dim; d++) {
        u[d] = (rand_01() - 0.5) * 50;
    }
    double p = (rand_01() + 1e-10) * 100;

    state[0] = rho;
    std::cout << "rho = " << rho << std::endl;
    double ke = 0.0;
    for (unsigned int d = 0; d < dim; d++) {
        state[d+1] = rho * u[d];
        ke += 0.5 * rho * u[d] * u[d];
    }
    state[dim+1] = ke + p / (gas_gamma - 1.0);

    return state;
}

