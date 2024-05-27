#include "src/five_moment/euler.h"
#include "src/five_moment/five_moment.h"
#include "src/five_moment/fluxes/subcell_finite_volume_flux.h"
#include "src/five_moment/solution_vec.h"
#include "src/warpii.h"
#include <gtest/gtest.h>

using namespace dealii;
using namespace warpii;

TEST(ShockCapturingFVTest, SingleCell1D) {
  char **argv = nullptr;
  int argc = 0;
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  Warpii warpii_obj;
  warpii_obj.input = R"(
set Application = FiveMoment
set n_dims = 1

set t_end = 0.01
set fields_enabled = false

set fe_degree = 2

subsection geometry
    set left = 0.0
    set right = 0.34
    set nx = 1
end
subsection Species_1
    subsection InitialCondition
        set Function constants = pi=3.1415926535
        set Function expression = 1 + 0.6 * sin(0.2*pi*(x)); \
                                  1 + 0.6 * sin(0.2*pi*(x)); \
                                  0.5 * (1 + 0.6*sin(0.2*pi*(x))) + 1.5
    end
end
)";
  warpii_obj.setup();

  auto &app = warpii_obj.get_app<five_moment::FiveMomentApp<1>>();
  // auto& op = app.get_solver().get_fluid_flux_operator();

  five_moment::FiveMSolutionVec dest;
  dest.reinit(app.get_solution());

  auto &mf = app.get_discretization().get_matrix_free();

  FEEvaluation<1, -1, 0, 3, double> phi(mf, 0, 1, 0);
  FEEvaluation<1, -1, 0, 3, double> phi_reader(mf, 0, 1, 0);

  phi.reinit(0);
  phi_reader.reinit(0);
  phi_reader.gather_evaluate(app.get_solution().mesh_sol,
                             EvaluationFlags::values);

  const double gamma = 5.0 / 3.0;
  auto flux =
      five_moment::SubcellFiniteVolumeFlux<1>(app.get_discretization(), gamma);

  flux.calculate_flux(dest.mesh_sol, phi, phi_reader, 1.0, false);

  phi.gather_evaluate(dest.mesh_sol, EvaluationFlags::values);
  // const auto& quad = mf.get_quadrature(0);
  // const auto& w = quad.get_weights();

  const double diff0_actual = phi.get_value(0)[0][0];

  Tensor<1, 1, VectorizedArray<double>> x_normal;
  x_normal[0] = VectorizedArray(1.0);

  double diff0_expected = 0.0;

  const auto subcell_area = phi.JxW(0)[0];
  const auto subcell_face_area = 1.0;

  const auto flux_in_left =
      five_moment::euler_flux<1>(phi_reader.get_value(0), gamma) * (x_normal);
  diff0_expected += (flux_in_left[0][0] * subcell_face_area) / subcell_area;

  const auto flux_out_right = five_moment::euler_CH_entropy_dissipating_flux(
      phi_reader.get_value(0), phi_reader.get_value(1), x_normal, gamma);
  diff0_expected -= (flux_out_right[0][0] * subcell_face_area) / subcell_area;

  // mimic integration step which will be reversed by inv mass matrix
  // multiplication
  diff0_expected *= phi.JxW(0)[0];

  EXPECT_NEAR(diff0_actual, diff0_expected, 1e-14);
}

TEST(ShockCapturingFVTest, SingleCell2D) {
  char **argv = nullptr;
  int argc = 0;
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  Warpii warpii_obj;
  warpii_obj.input = R"(
set Application = FiveMoment
set n_dims = 2

set t_end = 0.01
set fields_enabled = false

set fe_degree = 2

subsection geometry
    set left = 0.0,0.0
    set right = 0.34,0.27
    set nx = 1,1
end
subsection Species_1
    subsection InitialCondition
        set Function constants = pi=3.1415926535
        set Function expression = 1 + 0.6 * sin(0.2*pi*(x+y)); \
                                  1 + 0.6 * sin(0.2*pi*(x+y)); \
                                  1 + 0.6 * sin(0.2*pi*(x+y)); \
                                  0.5 * (1 + 0.6*sin(0.2*pi*(x+y))) + 1.5
    end
end
)";
  warpii_obj.setup();

  auto &app = warpii_obj.get_app<five_moment::FiveMomentApp<2>>();
  // auto& op = app.get_solver().get_fluid_flux_operator();

  five_moment::FiveMSolutionVec dest;
  dest.reinit(app.get_solution());

  auto &mf = app.get_discretization().get_matrix_free();

  FEEvaluation<2, -1, 0, 4, double> phi(mf, 0, 1, 0);
  FEEvaluation<2, -1, 0, 4, double> phi_reader(mf, 0, 1, 0);

  phi.reinit(0);
  phi_reader.reinit(0);
  phi_reader.gather_evaluate(app.get_solution().mesh_sol,
                             EvaluationFlags::values);

  const double gamma = 5.0 / 3.0;
  auto flux =
      five_moment::SubcellFiniteVolumeFlux<2>(app.get_discretization(), gamma);

  flux.calculate_flux(dest.mesh_sol, phi, phi_reader, 1.0, true);

  phi.gather_evaluate(dest.mesh_sol, EvaluationFlags::values);
  const auto &quad = QGaussLobatto<1>(3);
  const auto &w = quad.get_weights();

  const double diff0_actual = phi.get_value(0)[0][0];

  Tensor<1, 2, VectorizedArray<double>> x_normal;
  x_normal[0] = VectorizedArray(1.0);
  x_normal[1] = VectorizedArray(0.0);
  Tensor<1, 2, VectorizedArray<double>> y_normal;
  y_normal[0] = VectorizedArray(0.0);
  y_normal[1] = VectorizedArray(1.0);

  double diff0_expected = 0.0;

  const auto subcell_area = phi.JxW(0)[0];
  const auto x_subcell_face_area = .27 * w[0];
  const auto y_subcell_face_area = .34 * w[0];

  const auto flux_in_left =
      five_moment::euler_flux<2>(phi_reader.get_value(0), gamma) * (x_normal);
  diff0_expected += (flux_in_left[0][0] * x_subcell_face_area) / subcell_area;

  const auto flux_out_right = five_moment::euler_CH_entropy_dissipating_flux(
      phi_reader.get_value(0), phi_reader.get_value(1), x_normal, gamma);
  diff0_expected -= (flux_out_right[0][0] * x_subcell_face_area) / subcell_area;

  const auto flux_out_top = five_moment::euler_CH_entropy_dissipating_flux(
          phi_reader.get_value(0), phi_reader.get_value(3), y_normal, gamma);
  diff0_expected -= (flux_out_top[0][0] * y_subcell_face_area) / subcell_area;

  const auto flux_in_bottom = five_moment::euler_flux<2>(phi_reader.get_value(0), gamma) * (y_normal);
  diff0_expected += (flux_in_bottom[0][0] * y_subcell_face_area) / subcell_area;

  // mimic integration step which will be reversed by inv mass matrix
  // multiplication
  diff0_expected *= phi.JxW(0)[0];

  EXPECT_NEAR(diff0_actual, diff0_expected, 1e-15);
}
