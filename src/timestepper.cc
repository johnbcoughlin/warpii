#include "timestepper.h"
#include <iostream>
#include <cmath>

namespace warpii {
    void advance(
            std::function<bool(double t, double dt)> step,
            double t_end,
            std::function<double()> recommend_dt,
            std::vector<TimestepCallback>& callbacks) {
        double t = 0.0;
        double dt;

        std::vector<double> callback_times;
        for (auto &cb : callbacks) {
            if (cb.perform_zeroth) {
                cb.callback(0.0);
            }
            callback_times.push_back(t + cb.interval);
        }

        while (t < t_end - 1e-12) {
            auto next_callback_index = std::distance(callback_times.begin(),
                    std::min_element(callback_times.begin(), callback_times.end()));
            double next_callback_time = callbacks.empty() ? t_end : callback_times.at(next_callback_index);

            // Should we perform the callback when we get to the next stop,
            // or is that the end of the simulation with no callback scheduled
            bool perform_cb = next_callback_time < t_end && fabs(next_callback_time - t_end) > 1e-12;
            double next_stop = std::fmin(next_callback_time, t_end);

            // Put 1e-12 of slosh into the timestepping so we don't accidentally
            // step over by a whole dt, or take a "stutter step" of 1e-13.
            while (t < next_stop - 1e-12) {
                dt = fmin(recommend_dt(), next_stop - t);
                std::cout << "dt = " << dt << std::endl;
                bool succeeded = step(t, dt);
                if (!succeeded) {
                    continue;
                }

                t += dt;

                std::cout << "t + dt = " << t << std::endl;
            }

            if (perform_cb) {
                auto& cb = callbacks.at(next_callback_index);
                cb.callback(t);
                callback_times.at(next_callback_index) = t + cb.interval;
            }
        }
        for (unsigned int i = 0; i < callbacks.size(); i++) {
            auto& cb = callbacks.at(i);
            if (fabs(t_end - callback_times.at(i)) < 1e-12 || cb.perform_final) {
                cb.callback(t_end);
            }
        }
    }
}
