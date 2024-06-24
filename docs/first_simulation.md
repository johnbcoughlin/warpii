# Running a simple simulation {#first_simulation}

In this tutorial we run a simple single-species, five-moment simulation.
The equations we're solving are also called the compressible Euler equations.

WarpII uses deal.ii's [ParameterHandler](https://dealii.org/developer/doxygen/deal.II/classParameterHandler.html) class for user input.
The following example input initializes a single-fluid five-moment (Euler) simulation in one dimension.

`sine_wave.inp`:
```
set Application = FiveMoment
set n_dims = 1
set t_end = 1.0
set fields_enabled = false

# Use degree 2 finite elements, i.e. quadratic shape functions.
set fe_degree = 2

subsection geometry
    set left = 0.0
    set right = 1.0
    set nx = 50
end

subsection Species_1
    subsection InitialCondition
        # We will specify primitive variables: [rho, u_x, u_y, u_z, p].
        set VariablesType = Primitive
        set Function constants = pi=3.1415926535

        # The initial condition is specified using a parsed function.
        # The function components are separated by semicolons.
        set Function expression = 1 + 0.6 * sin(2*pi*x); 1.0; 0.0; 0.0; 1.0
    end
end
```

To run the simulation,
```
$ warpii sine_wave.inp
```
WarpII creates a directory to run the simulation and output files into:
```
$ ls FiveMoment__sine_wave/
solution_000.vtu
...
solution_010.vtu
```
