#include "src/five_moment/euler.h"
#include "euler_test_helpers.h"
#include "src/utilities.h"

#include <gtest/gtest.h>


using namespace warpii::five_moment;

TEST(EulerTest, EntropyVariablesSatisfyDerivativeIdentity) {
    double gamma = 1.4;
    Tensor<1, 5, double> q;
    q[0] = 2.3;
    q[1] = 0.58;
    q[2] = 0.33;
    q[3] = 0.0;
    q[4] = 10.0;

    const auto w = euler_entropy_variables<3>(q, gamma);
    //const auto eta = euler_mathematical_entropy<2>(q, gamma);

    // Perform a finite difference test of the relationship w = 
    double h = 1e-5;
    for (unsigned int i = 0; i < 4; i++) {
        auto q_L = Tensor(q);
        q_L[i] -= h;
        const double eta_L = euler_mathematical_entropy<2>(q_L, gamma);
        auto q_R = Tensor(q);
        q_R[i] += h;
        const double eta_R = euler_mathematical_entropy<2>(q_R, gamma);

        const double d_eta_di = (eta_R - eta_L) / (2*h);
        EXPECT_NEAR(d_eta_di, w[i], 1e-5);
    }
}

/**
 * Tests the identity psi = rho*u, which is eqn (4.5) from Chandrashekar (2012).
 */
TEST(EulerTest, EulerEntropyFluxPotentialTest) {
    double gamma = 1.4;
    random_euler_state<1>(gamma);
    for (unsigned int i = 0; i < 100; i++) {
        auto stateL = random_euler_state<1>(gamma);
        auto wL = euler_entropy_variables<1>(stateL, gamma);
        auto fL = euler_flux<1>(stateL, gamma);
        auto qL = euler_entropy_flux<1>(stateL, gamma);

        // Entropy flux potential
        Tensor<1, 3, double> psiL;
        for (unsigned int d = 0; d < 1; d++) {
            for (unsigned int c = 0; c < 5; c++) {
                psiL[d] += wL[c] * fL[c][d];
            }
            psiL[d] -= qL[d];
        }

        EXPECT_NEAR(psiL[0], stateL[1], 1e-8 * std::abs(stateL[0]));
    }
}

double ln_avg_reference_impl(double a, double b) {
    if (std::abs(a - b) < std::max(a, b) * 1e-6) {
        return (a + b) / 2.0;
    } else {
        return (b - a) / (std::log(b) - std::log(a));
    }
}

TEST(EulerFluxTests, LnAvgTest) {
    // Equal values
    EXPECT_EQ(ln_avg(0.4, 0.4), 0.4);
    EXPECT_NEAR(ln_avg(0.4, 0.6), ln_avg_reference_impl(0.4, 0.6), 1e-12);

    // Very small
    EXPECT_NEAR(ln_avg(1e-10, 1e-12), 2.1497576854210972e-11, 1e-16);
    EXPECT_NEAR(ln_avg(0.4, 0.4 + 1e-8), (0.8 + 1e-8) / 2.0, 1e-16);

    EXPECT_NEAR(ln_avg(1.0, 0.5), 0.7213475204444817, 1e-15);
}

TEST(EulerFluxTests, EulerCHECTest) {
    double gamma = 5.0 / 3.0;
    // Sod shocktube left and right states
    Tensor<1, 5, double> left({1.0, 0.0, 0.0, 0.0, 1.0 / (gamma - 1.0)});
    Tensor<1, 5, double> right({0.1, 0.0, 0.0, 0.0, 0.125 / (gamma - 1.0)});

    Tensor<1, 5, Tensor<1, 1, double>> actual = euler_CH_EC_flux<1>(left, right, gamma);
    EXPECT_EQ(actual[0][0], 0.0);
    EXPECT_NEAR(actual[1][0], 0.5 + 1.0 / 9, 1e-15);
    EXPECT_EQ(actual[4][0], 0.0);

    // Left: rho, u, p = [1.0, 1.0, 1.0]
    // Right: rho, u, p = [0.5, 0.5, 0.5]
    left = Tensor<1, 5, double>({1.0, 1.0, 0.0, 0.0, 0.5 * 1.0 + 1.0 / (gamma - 1.0)});
    right = Tensor<1, 5, double>({0.5, 0.5, 0.0, 0.0, 0.5 * 0.5 + 0.5 / (gamma - 1.0)});
    actual = euler_CH_EC_flux<1>(left, right, gamma);
    EXPECT_NEAR(actual[0][0], 0.7213475204444817, 1e-15);
    EXPECT_NEAR(actual[1][0],  1.4713475204444817, 1e-15);
    EXPECT_NEAR(actual[4][0], 2.192695040888963, 1e-15);
}

TEST(EulerFluxTests, EulerCHESTest) {
    double gamma = 5.0 / 3.0;
    // Sod shocktube left and right states
    Tensor<1, 5, double> left({1.0, 0.0, 0.0, 0.0, 1.0 / (gamma - 1.0)});
    Tensor<1, 5, double> right({0.1, 0.0, 0.0, 0.0, 0.125 / (gamma - 1.0)});

    Tensor<1, 1, double> normal;
    normal[0] = 1.0;
    Tensor<1, 5, double> actual = euler_CH_entropy_dissipating_flux<1>(
            left, right, normal, gamma);
    EXPECT_NEAR(actual[0], 0.6495190528383291, 1e-15);
    EXPECT_NEAR(actual[1], 0.5 + 1.0 / 9, 1e-15);
    EXPECT_NEAR(actual[4],  0.9381717944489488, 1e-15);
}

// This tests that the numerical entropy production,
// eqn (B.56) in Hennemann et al., is nearly zero.
TEST(EulerFluxTests, EntropyConservationTest1D) {
    double gamma = 1.4;
    random_euler_state<1>(gamma);
    for (unsigned int i = 0; i < 100; i++) {
        auto stateL = random_euler_state<1>(gamma);
        auto stateR = random_euler_state<1>(gamma);

        auto wL = euler_entropy_variables<1>(stateL, gamma);
        auto wR = euler_entropy_variables<1>(stateR, gamma);

        auto fL = euler_flux<1>(stateL, gamma);
        auto fR = euler_flux<1>(stateR, gamma);

        auto qL = euler_entropy_flux<1>(stateL, gamma);
        auto qR = euler_entropy_flux<1>(stateR, gamma);

        // Entropy flux potential
        Tensor<1, 1, double> psiL;
        Tensor<1, 1, double> psiR;
        for (unsigned int d = 0; d < 1; d++) {
            psiL[d] = 0.0;
            psiR[d] = 0.0;
            for (unsigned int c = 0; c < 5; c++) {
                psiL[d] += wL[c] * fL[c][d];
                psiR[d] += wR[c] * fR[c][d];
            }
            psiL[d] -= qL[d];
            psiR[d] -= qR[d];

            EXPECT_NEAR(psiL[d], stateL[d+1], 1e-10*stateL.norm());
            EXPECT_NEAR(psiR[d], stateR[d+1], 1e-10*stateR.norm());
        }

        Tensor<1, 1, double> n;
        n[0] = 1.0;

        const auto wjump = wR - wL;
        const auto psijump = psiR - psiL;
        const auto numerical_flux = euler_CH_EC_flux<1>(stateL, stateR, gamma) * n;

        const double r = wjump * numerical_flux - psijump * n;
        EXPECT_NEAR(r, 0.0, 1e-10 * (qL.norm() + qR.norm()));
    }
}
// This tests that the numerical entropy production,
// eqn (B.56) in Hennemann et al., is nearly zero.
TEST(EulerFluxTests, EntropyConservationTest2D) {
    double gamma = 1.4;
    random_euler_state<2>(gamma);
    for (unsigned int i = 0; i < 100; i++) {
        auto stateL = random_euler_state<2>(gamma);
        auto stateR = random_euler_state<2>(gamma);
        SHOW(stateL);
        SHOW(stateR);

        auto wL = euler_entropy_variables<2>(stateL, gamma);
        auto wR = euler_entropy_variables<2>(stateR, gamma);
        SHOW(wL);
        SHOW(wR);

        auto fL = euler_flux<2>(stateL, gamma);
        auto fR = euler_flux<2>(stateR, gamma);

        auto qL = euler_entropy_flux<2>(stateL, gamma);
        SHOW(qL);
        auto qR = euler_entropy_flux<2>(stateR, gamma);
        SHOW(qR);

        // Entropy flux potential
        Tensor<1, 2, double> psiL;
        Tensor<1, 2, double> psiR;
        for (unsigned int d = 0; d < 2; d++) {
            psiL[d] = 0.0;
            psiR[d] = 0.0;
            for (unsigned int c = 0; c < 5; c++) {
                psiL[d] += wL[c] * fL[c][d];
                psiR[d] += wR[c] * fR[c][d];
            }
            psiL[d] -= qL[d];
            psiR[d] -= qR[d];

            EXPECT_NEAR(psiL[d], stateL[d+1], 1e-10*stateL.norm());
            EXPECT_NEAR(psiR[d], stateR[d+1], 1e-10*stateR.norm());
        }

        SHOW(psiL);
        SHOW(psiR);

        // Normal direction
        Tensor<1, 2, double> n;
        n[0] = rand_01() + 0.1;
        n[1] = rand_01() + 0.1;
        n = n / n.norm();
        EXPECT_NEAR(n*n, 1.0, 1e-15);

        const auto wjump = wR - wL;
        const auto psijump = psiR - psiL;
        const auto numerical_flux = euler_CH_EC_flux<2>(stateL, stateR, gamma) * n;

        const double r = wjump * numerical_flux - n * psijump;
        EXPECT_NEAR(r, 0.0, 1e-10 * (qL.norm() + qR.norm()));
    }
}

// This tests that the numerical entropy production,
// eqn (B.56) in Hennemann et al., is negative for the entropy dissipating flux.
TEST(EulerFluxTests, EntropyDissipationTest2D) {
    double gamma = 1.4;
    random_euler_state<2>(gamma);
    for (unsigned int i = 0; i < 100; i++) {
        auto stateL = random_euler_state<2>(gamma);
        auto stateR = random_euler_state<2>(gamma);
        SHOW(stateL);
        SHOW(stateR);

        auto wL = euler_entropy_variables<2>(stateL, gamma);
        auto wR = euler_entropy_variables<2>(stateR, gamma);
        SHOW(wL);
        SHOW(wR);

        auto fL = euler_flux<2>(stateL, gamma);
        auto fR = euler_flux<2>(stateR, gamma);

        auto qL = euler_entropy_flux<2>(stateL, gamma);
        SHOW(qL);
        auto qR = euler_entropy_flux<2>(stateR, gamma);
        SHOW(qR);

        // Entropy flux potential
        Tensor<1, 2, double> psiL;
        Tensor<1, 2, double> psiR;
        for (unsigned int d = 0; d < 2; d++) {
            psiL[d] = 0.0;
            psiR[d] = 0.0;
            for (unsigned int c = 0; c < 5; c++) {
                psiL[d] += wL[c] * fL[c][d];
                psiR[d] += wR[c] * fR[c][d];
            }
            psiL[d] -= qL[d];
            psiR[d] -= qR[d];

            EXPECT_NEAR(psiL[d], stateL[d+1], 1e-10*stateL.norm());
            EXPECT_NEAR(psiR[d], stateR[d+1], 1e-10*stateR.norm());
        }

        SHOW(psiL);
        SHOW(psiR);

        // Normal direction
        Tensor<1, 2, double> n;
        n[0] = rand_01() + 0.1;
        n[1] = rand_01() + 0.1;
        n = n / n.norm();
        EXPECT_NEAR(n*n, 1.0, 1e-15);

        const auto wjump = wR - wL;
        const auto psijump = psiR - psiL;
        const auto numerical_flux = euler_CH_entropy_dissipating_flux<2>(stateL, stateR, n, gamma);

        const double r = wjump * numerical_flux - n * psijump;
        EXPECT_LE(r, 0.0);
    }
}
