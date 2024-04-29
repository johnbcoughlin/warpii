#pragma once
#include <functional>

namespace warpii {
    /**
     * Represents an operation to be performed at a fixed interval,
     * such as performing a solution writeout, logging, or computing diagnostics.
     */
    struct TimestepCallback {
        TimestepCallback(double interval, std::function<void(double)> callback,
                bool perform_zeroth = true, bool perform_final = true) :
            interval(interval), callback(callback), 
            perform_zeroth(perform_zeroth),
            perform_final(perform_final) {}
        double interval;
        std::function<void(double t)> callback;
        // Whether we should perform this callback at t = 0.
        bool perform_zeroth;
        // Whether we should perform this callback at t_end even if it's not scheduled.
        bool perform_final;
    };

    void advance(
            std::function<bool(double t, double dt)> step,
            double t_end,
            std::function<double()> recommend_dt,
            std::vector<TimestepCallback>& callbacks
            );
}
