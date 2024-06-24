#include <deal.II/base/tensor.h>

using namespace dealii;

// Returns a random real number between 0 and 1
double rand_01(); 

template <int dim>
Tensor<1, 5, double> random_euler_state(double gas_gamma) {
    Tensor<1, 5, double> state;
    Tensor<1, 3, double> u;
    double rho = (rand_01() + 1e-10) * 100;
    for (unsigned int d = 0; d < 3; d++) {
        u[d] = (rand_01() - 0.5) * 50;
    }
    double p = (rand_01() + 1e-10) * 100;

    state[0] = rho;
    double ke = 0.0;
    for (unsigned int d = 0; d < 3; d++) {
        state[d+1] = rho * u[d];
        ke += 0.5 * rho * u[d] * u[d];
    }
    state[4] = ke + p / (gas_gamma - 1.0);

    return state;
}

