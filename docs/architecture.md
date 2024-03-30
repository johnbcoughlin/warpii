# WarpII Architecture

At the moment, the architecture is very simple:
- The `codes` directory contains physics applications organized around specific equation sets:
    - `CartesianEuler`: the Euler equations in Cartesian geometry.
    - `CylindricalEuler`: the Euler equations in cylindrical geometry, with source terms eliminated by multiplying through by `r`.
- Each code is accompanied by one or more scripts, written in C++. A script is a file with a `main` function that can be built and executed as a main entrypoint.
- The `src` folder contains code shared between apps, such as Runge-Kutta timestepping methods.
