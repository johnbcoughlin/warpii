# Running a simple simulation {#first_simulation}

WarpII uses deal.ii's [ParameterHandler](https://dealii.org/developer/doxygen/deal.II/classParameterHandler.html) class for user input.
The following example input initializes a single-fluid five-moment (Euler) simulation
in one dimension.

`sine_wave_euler.inp`:
```
set Application = FiveMoment
set n_dims = 1
set t_end = 1.0
set fields_enabled = false

set fe_degree = 2

subsection geometry
    set left = 0.0
    set right = 1.0
    set nx = 10
end

subsection Species_1
    subsection InitialCondition
        set Function constants = pi=3.1415926535
        set Function expression = 1 + 0.6 * sin(2*pi*x); \
                                  1 + 0.6 * sin(2*pi*x); \
                                  0.5 * (1 + 0.6*sin(2*pi*x)) + 1.5
    end
end
```

To run the simulation,
```
mkdir -p workdir
cp sine_wave_euler.inp workdir
cd workdir
warpii sine_wave_euler.inp
```
Here we create a working directory by hand for WarpII.
WarpII always uses the directory it is invoked from as the working directory where it writes
out data files, log files, and meshes.
