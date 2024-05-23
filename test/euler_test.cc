#include "src/five_moment/euler.h"

#include <gtest/gtest.h>

using namespace warpii::five_moment;

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
    Tensor<1, 3, double> left({1.0, 0.0, 1.0 / (gamma - 1.0)});
    Tensor<1, 3, double> right({0.1, 0.0, 0.125 / (gamma - 1.0)});

    Tensor<1, 3, Tensor<1, 1, double>> actual = euler_CH_EC_flux<1>(left, right, gamma);
    EXPECT_EQ(actual[0][0], 0.0);
    EXPECT_NEAR(actual[1][0], 0.5 + 1.0 / 9, 1e-15);
    EXPECT_EQ(actual[2][0], 0.0);

    // Left: rho, u, p = [1.0, 1.0, 1.0]
    // Right: rho, u, p = [0.5, 0.5, 0.5]
    left = Tensor<1, 3, double>({1.0, 1.0, 0.5 * 1.0 + 1.0 / (gamma - 1.0)});
    right = Tensor<1, 3, double>({0.5, 0.5, 0.5 * 0.5 + 0.5 / (gamma - 1.0)});
    actual = euler_CH_EC_flux<1>(left, right, gamma);
    EXPECT_NEAR(actual[0][0], 0.7213475204444817, 1e-15);
    EXPECT_NEAR(actual[1][0],  1.4713475204444817, 1e-15);
    EXPECT_NEAR(actual[2][0], 2.192695040888963, 1e-15);
}

TEST(EulerFluxTests, EulerCHESTest) {
    double gamma = 5.0 / 3.0;
    // Sod shocktube left and right states
    Tensor<1, 3, double> left({1.0, 0.0, 1.0 / (gamma - 1.0)});
    Tensor<1, 3, double> right({0.1, 0.0, 0.125 / (gamma - 1.0)});

    Tensor<1, 1, double> normal;
    normal[0] = 1.0;
    Tensor<1, 3, double> actual = euler_CH_entropy_dissipating_flux<1>(
            left, right, normal, gamma);
    EXPECT_NEAR(actual[0], 0.6495190528383291, 1e-15);
    EXPECT_NEAR(actual[1], 0.5 + 1.0 / 9, 1e-15);
    EXPECT_NEAR(actual[2],  0.9381717944489488, 1e-15);
}
