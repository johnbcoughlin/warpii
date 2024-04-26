#include "src/timestepper.h"

#include <gtest/gtest.h>
#include <vector>

using namespace warpii;

TEST(TimestepperTest, NoCallbacks) {
    auto step = [](double, double) -> bool { return true; };
    auto recommend_dt = []() -> double { return 0.024; };
    std::vector<TimestepCallback> callbacks;

    ASSERT_NO_THROW(advance(step, 19.0, recommend_dt, callbacks));
}

TEST(TimestepperTest, StopsAtCallbacks) {
    auto step = [](double, double) -> bool { return true; };
    auto recommend_dt = []() -> double { return 0.024; };

    std::vector<double> writeout_times;
    auto writeout = TimestepCallback(
        0.3, [&](double t) -> void { writeout_times.push_back(t); });

    std::vector<double> diagnostic_times;
    auto diagnostic = TimestepCallback(
            0.1, [&](double t) -> void { diagnostic_times.push_back(t); });

    std::vector<double> print_profile_times;
    auto print_profile = TimestepCallback(
            0.25, [&](double t) -> void { print_profile_times.push_back(t); },
            false, true);

    std::vector<TimestepCallback> callbacks = { writeout, diagnostic, print_profile };

    advance(step, 1.2, recommend_dt, callbacks);

    // Expected writeouts: 0.0, 0.3, 0.6, 0.9, 1.2
    ASSERT_EQ(writeout_times.size(), 5);
    ASSERT_EQ(writeout_times.at(0), 0.0);
    ASSERT_NEAR(writeout_times.at(3), 0.9, 1e-12);
    ASSERT_NEAR(writeout_times.at(4), 1.2, 1e-12);

    // Expected diagnostics: 0.0, 0.1, 0.2, 0.3, ...., 1.1, 1.2
    ASSERT_EQ(diagnostic_times.size(), 13);
    ASSERT_EQ(diagnostic_times.at(0), 0.0);
    ASSERT_NEAR(diagnostic_times.at(6), 0.6, 1e-12);
    ASSERT_NEAR(diagnostic_times.at(12), 1.2, 1e-12);

    ASSERT_EQ(print_profile_times.size(), 5);
    ASSERT_NEAR(print_profile_times.at(0), 0.25, 1e-12);
    ASSERT_NEAR(print_profile_times.at(2), 0.75, 1e-12);
    ASSERT_NEAR(print_profile_times.at(4), 1.2, 1e-12);
}

TEST(TimestepperTest, NoWastedSteps) {
    unsigned int step_count = 0;
    auto step = [&step_count](double, double) -> bool { 
        step_count++;
        return true; 
    };
    auto recommend_dt = []() -> double { return 0.024; };

    std::vector<double> writeout_times;
    auto writeout = TimestepCallback(
        0.3, [&](double t) -> void { writeout_times.push_back(t); });
    std::vector<double> diagnostic_times;
    auto diagnostic = TimestepCallback(
            0.3, [&](double t) -> void { diagnostic_times.push_back(t); });

    std::vector<TimestepCallback> callbacks = { writeout, diagnostic };
    advance(step, 1.2, recommend_dt, callbacks);
    unsigned int step_count_two_cbs = step_count;
    step_count = 0;

    callbacks = { writeout };
    advance(step, 1.2, recommend_dt, callbacks);
    unsigned int step_count_one_cb = step_count;

    ASSERT_EQ(step_count_two_cbs, 52);
    ASSERT_EQ(step_count_one_cb, 52);
}
