#include "fpe.h"

#include <iostream>
#ifdef __linux__
#include <cfenv>
#elif __APPLE__

#ifdef __aarch64__
// ARM fix
#include <fenv.h>
#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#endif
#endif

#ifdef __linux__
void enable_floating_point_exceptions() {
    // kills sim if there appears nan (on linux)
    feenableexcept(FE_INVALID);
}
#elif __APPLE__
#if defined(__aarch64__)
// see
// https://stackoverflow.com/questions/69059981/how-to-trap-floating-point-exceptions-on-m1-macs
static void fpe_signal_handler(int, siginfo_t* sip, void*) {
    int fe_code = sip->si_code;

    std::cerr << "In signal handler : ";

    if (fe_code == ILL_ILLTRP)
        std::cerr << "Floating point exception" << std::endl;
    else
        std::cerr << "Code detected : " << fe_code << std::endl;

    abort();
}

void enable_floating_point_exceptions() {
    fenv_t env;
    fegetenv(&env);

    env.__fpcr = env.__fpcr | __fpcr_trap_invalid;
    fesetenv(&env);

    struct sigaction act;
    act.sa_sigaction = fpe_signal_handler;
    sigemptyset(&act.sa_mask);
    act.sa_flags = SA_SIGINFO;
    sigaction(SIGILL, &act, NULL);
}
#endif
#endif
