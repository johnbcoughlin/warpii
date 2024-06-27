#include <deal.II/base/utilities.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/vector.h>
#include <deal.II/matrix_free/evaluation_flags.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>

#include <vector>

#include "../dof_utils.h"
#include "../nodal_dg/nodal_dg_discretization.h"

using namespace dealii;

namespace warpii {

template <int dim>
class MaxwellFluxDGOperator {
    void perform_forward_euler_step(
        LinearAlgebra::distributed::Vector<double> &dst,
        const LinearAlgebra::distributed::Vector<double> &u,
        std::vector<LinearAlgebra::distributed::Vector<double>> &sol_registers,
        const double dt, const double t, const double alpha = 1.0,
        const double beta = 0.0,
        const ZeroOutPolicy zero_out_policy = DO_NOT_ZERO_DST_VECTOR);

   private:
    std::shared_ptr<NodalDGDiscretization<dim>> discretization;
    unsigned int first_component_index;
    double omega_c_tau;
    double omega_p_tau;
    double chi;
    double gamma;

    void local_apply_inverse_mass_matrix(
        const MatrixFree<dim, double> &mf,
        LinearAlgebra::distributed::Vector<double> &dst,
        const LinearAlgebra::distributed::Vector<double> &src,
        const std::pair<unsigned int, unsigned int> &cell_range) const;

    void local_apply_cell(
        const MatrixFree<dim, double> &mf,
        LinearAlgebra::distributed::Vector<double> &dst,
        const LinearAlgebra::distributed::Vector<double> &src,
        const std::pair<unsigned int, unsigned int> &cell_range);

    void local_apply_face(
        const MatrixFree<dim, double> &mf,
        LinearAlgebra::distributed::Vector<double> &dst,
        const LinearAlgebra::distributed::Vector<double> &src,
        const std::pair<unsigned int, unsigned int> &face_range) const;

    void local_apply_boundary_face(
        const MatrixFree<dim, double> &mf,
        LinearAlgebra::distributed::Vector<double> &dst,
        const LinearAlgebra::distributed::Vector<double> &src,
        const std::pair<unsigned int, unsigned int> &face_range) const;
};

template <int dim>
void MaxwellFluxDGOperator<dim>::perform_forward_euler_step(
    LinearAlgebra::distributed::Vector<double> &dst,
    const LinearAlgebra::distributed::Vector<double> &u,
    std::vector<LinearAlgebra::distributed::Vector<double>> &sol_registers,
    const double dt, const double, const double alpha, const double beta,
    const ZeroOutPolicy zero_out_policy) {
    auto Mdudt_register = sol_registers.at(0);
    auto dudt_register = sol_registers.at(1);

    discretization->mf.loop(
        &MaxwellFluxDGOperator<dim>::local_apply_cell,
        &MaxwellFluxDGOperator<dim>::local_apply_face,
        &MaxwellFluxDGOperator<dim>::local_apply_boundary_face, this,
        dudt_register, u, zero_out_policy == DO_ZERO_DST_VECTOR,
        MatrixFree<dim, double>::DataAccessOnFaces::values,
        MatrixFree<dim, double>::DataAccessOnFaces::values);

    {
        discretization->mf.cell_loop(
            &MaxwellFluxDGOperator<dim>::local_apply_inverse_mass_matrix, this,
            dudt_register, Mdudt_register,
            std::function<void(const unsigned int, const unsigned int)>(),
            [&](const unsigned int start_range, const unsigned int end_range) {
                /* DEAL_II_OPENMP_SIMD_PRAGMA */
                for (unsigned int i = start_range; i < end_range; ++i) {
                    const double dudt_i = dudt_register.local_element(i);
                    const double dst_i = dst.local_element(i);
                    const double u_i = u.local_element(i);
                    dst.local_element(i) =
                        beta * dst_i + alpha * (u_i + dt * dudt_i);
                }
            });
    }
}

template <int dim>
void MaxwellFluxDGOperator<dim>::local_apply_inverse_mass_matrix(
    const MatrixFree<dim, double> &mf,
    LinearAlgebra::distributed::Vector<double> &dst,
    const LinearAlgebra::distributed::Vector<double> &src,
    const std::pair<unsigned int, unsigned int> &cell_range) const {
    unsigned int first_component = first_component_index;
    FEEvaluation<dim, -1, 0, 8, double> phi(mf, 0, 1, first_component);
    MatrixFreeOperators::CellwiseInverseMassMatrix<dim, -1, dim + 2, double>
        inverse(phi);

    for (unsigned int cell = cell_range.first; cell < cell_range.second;
         ++cell) {
        phi.reinit(cell);
        phi.read_dof_values(src);

        inverse.apply(phi.begin_dof_values(), phi.begin_dof_values());

        phi.set_dof_values(dst);
    }
}

template <int dim>
void MaxwellFluxDGOperator<dim>::local_apply_cell(
    const MatrixFree<dim, double> &mf,
    LinearAlgebra::distributed::Vector<double> &dst,
    const LinearAlgebra::distributed::Vector<double> &src,
    const std::pair<unsigned int, unsigned int> &cell_range) {
    FEEvaluation<dim, -1, 0, 8, double> fe_eval(mf, 0, 1,
                                                first_component_index);

    for (unsigned int cell = cell_range.first; cell < cell_range.second;
         cell++) {
        fe_eval.reinit(cell);
        fe_eval.gather_evaluate(src, EvaluationFlags::values);
        for (unsigned int q : fe_eval.quadrature_point_indices()) {
            const auto val = fe_eval.get_value(q);
            Tensor<1, 3, double> E;
            Tensor<1, 3, double> B;
            for (unsigned int d = 0; d < 3; d++) {
                E[d] = val[d];
                B[d] = val[d + 3];
            }
            const double phi = val[6];
            const double psi = val[7];

            Tensor<1, 8, Tensor<1, dim, VectorizedArray<double>>> flux;

            // x-direction first
            double c = omega_p_tau / omega_c_tau;
            double c2 = c * c;

            flux[0][0] = chi * c2 * phi;
            flux[1][0] = c2 * B[2];
            flux[2][0] = -c2 * B[1];
            flux[3][0] = gamma * psi;
            flux[4][0] = -E[2];
            flux[5][0] = E[1];
            flux[6][0] = chi * E[0];
            flux[7][0] = gamma * c2 * B[0];

            // y-direction
            if (dim > 1) {
                flux[0][1] = -c2 * B[2];
                flux[1][1] = chi * c2 * phi;
                flux[2][1] = c2 * B[0];
                flux[3][1] = E[2];
                flux[4][1] = gamma * psi;
                flux[5][1] = -E[0];
                flux[6][1] = chi * E[1];
                flux[7][1] = gamma * c2 * B[1];
            }
            if (dim > 2) {
                flux[0][2] = c2 * B[1];
                flux[1][2] = -c2 * B[0];
                flux[2][2] = chi * c2 * phi;
                flux[3][2] = -E[1];
                flux[4][2] = E[0];
                flux[5][2] = gamma * psi;
                flux[6][2] = chi * E[2];
                flux[7][2] = gamma * c2 * B[2];
            }
            fe_eval.submit_gradient(flux, q);
        }
        fe_eval.integrate_scatter(EvaluationFlags::gradients, dst);
    }
}

template <int dim>
void MaxwellFluxDGOperator<dim>::local_apply_face(
    const MatrixFree<dim, double> &mf,
    LinearAlgebra::distributed::Vector<double> &dst,
    const LinearAlgebra::distributed::Vector<double> &src,
    const std::pair<unsigned int, unsigned int> &face_range) const {
    FEFaceEvaluation<dim, -1, 0, dim + 2, double> fe_eval_m(
        mf, true, 0, 1, first_component_index);
    FEFaceEvaluation<dim, -1, 0, dim + 2, double> fe_eval_p(
        mf, false, 0, 1, first_component_index);
    for (unsigned int face = face_range.first; face < face_range.second;
         face++) {
        fe_eval_p.reinit(face);
        fe_eval_p.gather_evaluate(src, EvaluationFlags::values);

        fe_eval_m.reinit(face);
        fe_eval_m.gather_evaluate(src, EvaluationFlags::values);
        const double c = omega_p_tau / omega_c_tau;
        const double c2 = c * c;
        for (const unsigned int q : fe_eval_m.quadrature_point_indices()) {
            const auto n = fe_eval_m.normal_vector(q);

            const auto val_m = fe_eval_m.get_value(q);
            const auto val_p = fe_eval_p.get_value(q);

            Tensor<1, 3, VectorizedArray<double>> E_m;
            Tensor<1, 3, VectorizedArray<double>> B_m;
            Tensor<1, 3, VectorizedArray<double>> E_p;
            Tensor<1, 3, VectorizedArray<double>> B_p;
            for (unsigned int d = 0; d < 3; d++) {
                E_m[d] = val_m[d];
                B_m[d] = val_m[d + 3];
            }

            Tensor<1, 8, VectorizedArray<double>> numerical_flux;
            for (unsigned int comp = 0; comp < 8; comp++) {
                numerical_flux[comp] = 0.0;
            }

            const auto B_avg_cross_n = cross_product_3d(0.5 * (B_m + B_p), n);
            const auto B_jump_cross_n = cross_product_3d(B_p - B_m, n);
            const auto E_avg_cross_n = cross_product_3d(0.5 * (E_m + E_p), n);
            const auto E_jump_cross_n = cross_product_3d(E_p - E_m, n);

            const auto E_avg_dot_n = 0.5 * (E_m + E_p) * n;
            const auto B_avg_dot_n = 0.5 * (E_m + E_p) * n;
            const auto E_jump_dot_n = (E_p - E_m) * n;
            const auto B_jump_dot_n = (B_p - B_m) * n;

            const auto phi_avg = 0.5 * (val_m[6] + val_p[6]);
            const auto psi_avg = 0.5 * (val_m[7] + val_p[7]);
            const auto phi_jump = val_p[6] - val_m[6];
            const auto psi_jump = val_p[7] - val_m[7];

            const auto E_flux = -c2 * B_avg_cross_n -
                                0.5 * c * cross_product_3d(n, E_jump_cross_n) +
                                chi * c2 * phi_avg -
                                0.5 * chi * c * E_jump_dot_n;
            const auto B_flux =
                E_avg_cross_n - 0.5 * c * cross_product_3d(n, B_jump_cross_n) +
                gamma * c2 * psi_avg - 0.5 * gamma * c * B_jump_dot_n;

            const auto phi_flux = chi * E_avg_dot_n - 0.5 * chi * c * phi_jump;
            const auto psi_flux =
                gamma * c2 * B_avg_dot_n - 0.5 * gamma * c * psi_jump;
        }
    }
}

}  // namespace warpii
