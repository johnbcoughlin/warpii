#include <deal.II/base/types.h>
#include <deal.II/base/utilities.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/vector.h>
#include <deal.II/matrix_free/evaluation_flags.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>

#include <boost/mpl/pair.hpp>
#include <vector>

#include "../dgsem/nodal_dg_discretization.h"
#include "../dof_utils.h"
#include "../rk.h"
#include "../function_eval.h"
#include "fields.h"
#include "bc_map.h"
#include "maxwell.h"
#include "../geometry.h"

using namespace dealii;

namespace warpii {

template <int dim, typename SolutionVec>
class MaxwellFluxDGOperator : ForwardEulerOperator<SolutionVec> {
   public:
    MaxwellFluxDGOperator(
        std::shared_ptr<NodalDGDiscretization<dim>> discretization,
        unsigned int first_component_index, 
        std::shared_ptr<PHMaxwellFields<dim>> fields
        )
        : discretization(discretization),
          first_component_index(first_component_index),
          fields(fields),
          constants(fields->phmaxwell_constants())
    {}

    void perform_forward_euler_step(SolutionVec &dst, const SolutionVec &u,
                                    std::vector<SolutionVec> &sol_registers,
                                    const double dt, const double t,
                                    const double b, const double a,
                                    const double c) override;

   private:
    std::shared_ptr<NodalDGDiscretization<dim>> discretization;
    unsigned int first_component_index;
    std::shared_ptr<PHMaxwellFields<dim>> fields;
    PHMaxwellConstants constants;

    void local_apply_inverse_mass_matrix(
        const MatrixFree<dim, double> &mf,
        LinearAlgebra::distributed::Vector<double> &dst,
        const LinearAlgebra::distributed::Vector<double> &src,
        const std::pair<unsigned int, unsigned int> &cell_range) const;

    void local_apply_cell(
        const MatrixFree<dim, double> &mf,
        LinearAlgebra::distributed::Vector<double> &dst,
        const LinearAlgebra::distributed::Vector<double> &src,
        const std::pair<unsigned int, unsigned int> &cell_range) const;

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

template <int dim, typename SolutionVec>
void MaxwellFluxDGOperator<dim, SolutionVec>::perform_forward_euler_step(
    SolutionVec &dst, const SolutionVec &u,
    std::vector<SolutionVec> &sol_registers, const double dt, const double,
    const double b, const double a, const double c) {
    auto Mdudt_register = sol_registers.at(0);
    auto dudt_register = sol_registers.at(1);

    // bool zero_out_register = true;
    discretization->mf.loop(
        &MaxwellFluxDGOperator<dim, SolutionVec>::local_apply_cell,
        &MaxwellFluxDGOperator<dim, SolutionVec>::local_apply_face,
        &MaxwellFluxDGOperator<dim, SolutionVec>::local_apply_boundary_face,
        this, dudt_register.mesh_sol, u.mesh_sol, true,
        MatrixFree<dim, double>::DataAccessOnFaces::values,
        MatrixFree<dim, double>::DataAccessOnFaces::values);

    {
        discretization->mf.cell_loop(
            &MaxwellFluxDGOperator<
                dim, SolutionVec>::local_apply_inverse_mass_matrix,
            this, dudt_register.mesh_sol, Mdudt_register.mesh_sol,
            std::function<void(const unsigned int, const unsigned int)>(),
            [&](const unsigned int start_range, const unsigned int end_range) {
                /* DEAL_II_OPENMP_SIMD_PRAGMA */
                for (unsigned int i = start_range; i < end_range; ++i) {
                    const double dudt_i =
                        dudt_register.mesh_sol.local_element(i);
                    const double dst_i = dst.mesh_sol.local_element(i);
                    const double u_i = u.mesh_sol.local_element(i);
                    dst.mesh_sol.local_element(i) =
                        b * dst_i + a * u_i + c * dt * dudt_i;
                }
            });
    }
}

template <int dim, typename SolutionVec>
void MaxwellFluxDGOperator<dim, SolutionVec>::local_apply_inverse_mass_matrix(
    const MatrixFree<dim, double> &mf,
    LinearAlgebra::distributed::Vector<double> &dst,
    const LinearAlgebra::distributed::Vector<double> &src,
    const std::pair<unsigned int, unsigned int> &cell_range) const {
    unsigned int first_component = first_component_index;
    FEEvaluation<dim, -1, 0, 8, double> phi(mf, 0, 1, first_component);
    MatrixFreeOperators::CellwiseInverseMassMatrix<dim, -1, 8, double> inverse(
        phi);

    for (unsigned int cell = cell_range.first; cell < cell_range.second;
         ++cell) {
        phi.reinit(cell);
        phi.read_dof_values(src);

        inverse.apply(phi.begin_dof_values(), phi.begin_dof_values());

        phi.set_dof_values(dst);
    }
}

template <int dim, typename SolutionVec>
void MaxwellFluxDGOperator<dim, SolutionVec>::local_apply_cell(
    const MatrixFree<dim, double> &mf,
    LinearAlgebra::distributed::Vector<double> &dst,
    const LinearAlgebra::distributed::Vector<double> &src,
    const std::pair<unsigned int, unsigned int> &cell_range) const {
    FEEvaluation<dim, -1, 0, 8, double> fe_eval(mf, 0, 1,
                                                first_component_index);

    for (unsigned int cell = cell_range.first; cell < cell_range.second;
         cell++) {
        fe_eval.reinit(cell);
        fe_eval.gather_evaluate(src, EvaluationFlags::values);
        for (unsigned int q : fe_eval.quadrature_point_indices()) {
            const auto val = fe_eval.get_value(q);
            Tensor<1, 8, Tensor<1, dim, VectorizedArray<double>>> flux =
                ph_maxwell_flux<dim>(val, constants);
            fe_eval.submit_gradient(flux, q);
        }
        fe_eval.integrate_scatter(EvaluationFlags::gradients, dst);
    }
}

template <int dim, typename SolutionVec>
void MaxwellFluxDGOperator<dim, SolutionVec>::local_apply_face(
    const MatrixFree<dim, double> &mf,
    LinearAlgebra::distributed::Vector<double> &dst,
    const LinearAlgebra::distributed::Vector<double> &src,
    const std::pair<unsigned int, unsigned int> &face_range) const {
    FEFaceEvaluation<dim, -1, 0, 8, double> fe_eval_m(mf, true, 0, 1,
                                                      first_component_index);
    FEFaceEvaluation<dim, -1, 0, 8, double> fe_eval_p(mf, false, 0, 1,
                                                      first_component_index);
    for (unsigned int face = face_range.first; face < face_range.second;
         face++) {
        fe_eval_p.reinit(face);
        fe_eval_p.gather_evaluate(src, EvaluationFlags::values);

        fe_eval_m.reinit(face);
        fe_eval_m.gather_evaluate(src, EvaluationFlags::values);
        for (const unsigned int q : fe_eval_m.quadrature_point_indices()) {
            const auto n = fe_eval_m.normal_vector(q);

            const auto val_m = fe_eval_m.get_value(q);
            const auto val_p = fe_eval_p.get_value(q);

            Tensor<1, 8, VectorizedArray<double>> numerical_flux =
                ph_maxwell_numerical_flux<dim>(val_m, val_p, n, constants);

            fe_eval_m.submit_value(-numerical_flux, q);
            fe_eval_p.submit_value(numerical_flux, q);
        }

        fe_eval_m.integrate_scatter(EvaluationFlags::values, dst);
        fe_eval_p.integrate_scatter(EvaluationFlags::values, dst);
    }
}

template <int dim, typename SolutionVec>
void MaxwellFluxDGOperator<dim, SolutionVec>::local_apply_boundary_face(
    const MatrixFree<dim, double> &mf,
    LinearAlgebra::distributed::Vector<double> &dst,
    const LinearAlgebra::distributed::Vector<double> &src,
    const std::pair<unsigned int, unsigned int> &face_range) const {

    using VA = VectorizedArray<double>;

    FEFaceEvaluation<dim, -1, 0, 8, double> fe_eval_m(mf, true, 0, 1,
                                                      first_component_index);

    for (unsigned int face = face_range.first; face < face_range.second;
         face++) {
        fe_eval_m.reinit(face);
        fe_eval_m.gather_evaluate(src, EvaluationFlags::values);

        const types::boundary_id boundary_id = mf.get_boundary_id(face);
        MaxwellBCType bc_type = fields->get_bc_map().get_bc_type(boundary_id);

        for (const unsigned int q : fe_eval_m.quadrature_point_indices()) {
            const auto n = fe_eval_m.normal_vector(q);
            const Tensor<1, 3, VA> n3d = at_least_3d<dim, VA>(n);
            const auto val_m = fe_eval_m.get_value(q);
            //const auto t_and_b = tangent_and_binormal<VA>(n3d);

            Tensor<1, 3, VA> E_m;
            Tensor<1, 3, VA> B_m;
            VA phi_m;
            VA psi_m;
            for (unsigned int d = 0; d < 3; d++) {
                E_m = val_m[d];
                B_m = val_m[d+3];
            }
            phi_m = val_m[6];
            psi_m = val_m[7];

            Tensor<1, 3, VA> E_bdy;
            Tensor<1, 3, VA> B_bdy;
            VA phi_bdy;
            VA psi_bdy;
            if (bc_type == MaxwellBCType::PERFECT_CONDUCTOR) {
                // Just the normal component of E
                E_bdy = (n3d * E_m) * n3d;
                // Just the tangential components of B
                B_bdy = B_m - (n3d * B_m) * n3d;
                phi_bdy = VA(0.0);
                psi_bdy = VA(0.0);
            } else if (bc_type == MaxwellBCType::DIRICHLET) {
                const auto p = fe_eval_m.quadrature_point(q);
                const auto func = fields->get_bc_map().get_dirichlet_func(boundary_id);
                E_bdy = evaluate_function<dim, VA, 3>(*func->E_func, p);
                B_bdy = evaluate_function<dim, VA, 3>(*func->B_func, p);
                phi_bdy = evaluate_function<dim, double, VA>(*func->phi_func, p, 0);
                psi_bdy = evaluate_function<dim, double, VA>(*func->psi_func, p, 0);
            }

            Tensor<1, 8, VA> val_bdy;
            for (unsigned int d = 0; d < 3; d++) {
                val_bdy[d] = E_bdy[d];
                val_bdy[d+3] = B_bdy[d];
            }
            val_bdy[6] = phi_bdy;
            val_bdy[7] = psi_bdy;

            const Tensor<1, 8, VA> val_p = 2.0*val_bdy - val_m;

            const auto numerical_flux = ph_maxwell_numerical_flux(val_m, val_p, n, constants);
            fe_eval_m.submit_value(-numerical_flux, q);
        }

        fe_eval_m.integrate_scatter(EvaluationFlags::values, dst);
    }
}

}  // namespace warpii
