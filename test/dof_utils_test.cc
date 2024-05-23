#include <gtest/gtest.h>
#include "src/dof_utils.h"

using namespace warpii;

TEST(DOFUtilsTest, PencilBaseTest) {
    // 1D
    EXPECT_EQ(pencil_base<1>(7, 9, 0), 0);

    // 2D, x
    EXPECT_EQ(pencil_base<2>(3, 2, 0), 2);
    EXPECT_EQ(pencil_base<2>(2, 2, 0), 2);
    EXPECT_EQ(pencil_base<2>(7, 3, 0), 6);
    EXPECT_EQ(pencil_base<2>(8, 3, 0), 6);

    // 2D, y
    EXPECT_EQ(pencil_base<2>(3, 2, 1), 1);
    EXPECT_EQ(pencil_base<2>(6, 3, 1), 0);
    EXPECT_EQ(pencil_base<2>(5, 3, 1), 2);
}

TEST(DOFUtilsTest, QuadPointNeighborTest) {
    // 1D
    EXPECT_EQ(quadrature_point_neighbor<1>(7, 5, 9, 0), 5);

    // 2D, x
    EXPECT_EQ(quadrature_point_neighbor<2>(7, 0, 3, 0), 6);
    EXPECT_EQ(quadrature_point_neighbor<2>(7, 1, 3, 0), 7);
    EXPECT_EQ(quadrature_point_neighbor<2>(7, 2, 3, 0), 8);

    // 2D, y
    EXPECT_EQ(quadrature_point_neighbor<2>(7, 0, 3, 1), 1);
    EXPECT_EQ(quadrature_point_neighbor<2>(7, 1, 3, 1), 4);
    EXPECT_EQ(quadrature_point_neighbor<2>(7, 2, 3, 1), 7);
}

TEST(DOFUtilsTest, QuadPoint1DIndexTest) {
    // 1D
    EXPECT_EQ(quad_point_1d_index<1>(7, 9, 0), 7);

    // 2D, x
    EXPECT_EQ(quad_point_1d_index<2>(8, 3, 0), 2);
    EXPECT_EQ(quad_point_1d_index<2>(4, 3, 0), 1);

    // 2D, y
    EXPECT_EQ(quad_point_1d_index<2>(8, 3, 1), 2);
    EXPECT_EQ(quad_point_1d_index<2>(5, 3, 1), 1);
    EXPECT_EQ(quad_point_1d_index<2>(4, 3, 1), 1);
    EXPECT_EQ(quad_point_1d_index<2>(1, 3, 1), 0);
}

TEST(DOFUtilsTest, PencilStartsTest) {
    EXPECT_EQ(pencil_starts<1>(8, 0).size(), 1);
    EXPECT_EQ(pencil_starts<1>(8, 0)[0], 0);

    EXPECT_EQ(pencil_starts<2>(3, 0).size(), 3);
    EXPECT_EQ(pencil_starts<2>(3, 0)[1], 3);
    EXPECT_EQ(pencil_starts<2>(3, 0)[2], 6);

    EXPECT_EQ(pencil_starts<2>(3, 1).size(), 3);
    EXPECT_EQ(pencil_starts<2>(3, 1)[1], 1);
    EXPECT_EQ(pencil_starts<2>(3, 1)[2], 2);
}
