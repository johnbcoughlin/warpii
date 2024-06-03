#include <deal.II/base/mpi.h>
#include "src/five_moment/extension.h"

int main(int argc, char** argv) {
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
}
