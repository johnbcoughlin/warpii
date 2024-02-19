#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/vectorization.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/matrix_free/matrix_free.h>

namespace Abstract_DG {
using namespace dealii;
template <int dim, int degree, int n_points_1d, typename VectorizedNumber>
class DGMultiphysicsOperator {
   public:
    virtual int n_components() = 0;

    virtual void local_apply_cell(
        const MatrixFree<dim, VectorizedNumber> &,
        LinearAlgebra::distributed::Vector<VectorizedNumber> &dst,
        const LinearAlgebra::distributed::Vector<VectorizedNumber> &src,
        const std::pair<unsigned int, unsigned int> &cell_range) const = 0;

    virtual void local_apply_face(
        const MatrixFree<dim, VectorizedNumber> &,
        LinearAlgebra::distributed::Vector<VectorizedNumber> &dst,
        const LinearAlgebra::distributed::Vector<VectorizedNumber> &src,
        const std::pair<unsigned int, unsigned int> &cell_range) const = 0;

    virtual void local_apply_boundary_face(
        const MatrixFree<dim, VectorizedNumber> &mf,
        LinearAlgebra::distributed::Vector<VectorizedNumber> &dst,
        const LinearAlgebra::distributed::Vector<VectorizedNumber> &src,
        const std::pair<unsigned int, unsigned int> &face_range) const = 0;
};
}  // namespace Abstract_DG

namespace Euler_DG {
using namespace dealii;

using Number = double;

template <int dim>
class EulerProblem {
   public:
    EulerProblem();

    void run();

   private:
    void make_grid_and_dofs();
    LinearAlgebra::distributed::Vector<Number> solution;
    ConditionalOStream pcout;
};

template <int dim>
void EulerProblem<dim>::run() {
    {
        const unsigned int n_vect_number = VectorizedArray<Number>::size();
        const unsigned int n_vect_bits = 8 * sizeof(Number) * n_vect_number;

        pcout << "Running with "
              << Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)
              << " MPI processes" << std::endl;
        pcout << "Vectorization over " << n_vect_number << ' '
              << (std::is_same_v<Number, double> ? "doubles" : "floats")
              << " = " << n_vect_bits << " bits ("
              << Utilities::System::get_current_vectorization_level() << ')'
              << std::endl;
    }
    make_grid_and_dofs();
}

}  // namespace Euler_DG

int main(int argc, char **argv) {
    using namespace dealii;
    using namespace Euler_DG;

    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
}
