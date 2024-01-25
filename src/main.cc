#include "spec/SimulationSpec.h"
#include "DGOperator.h"
#include "DGOperatorFactories.h"

using namespace WarpII;

int main() {
    int dim = 2;
    int degree = 4;
    DGOperatorSpec dg_spec {EULER_DG_OPERATOR, degree};
    SimulationSpec sim_spec{
        MeshSpec{dim},
        VariableSpec{EULER_VARIABLES, NonMeshVariableSpec{}},
        RungeKuttaSpec{std::variant<DGOperatorSpec>(dg_spec)},
    };

    ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
    TimerOutput timer(pcout, TimerOutput::never, TimerOutput::wall_times);
    auto timerPtr = std::shared_ptr<TimerOutput>(&timer);
    //auto rk_spec = std::get<RungeKuttaSpec>(sim_spec.timestepping_spec);
    //DGOperatorSpec dg_spec = std::get<DGOperatorSpec>(rk_spec.mesh_operator);
    auto dg = createDGOperator<double>(dim, degree, dg_spec, {timerPtr});
}
