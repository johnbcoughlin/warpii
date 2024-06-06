# A more complex example: extending WarpII with C++ {#extension_tutorial}

The input file format used by WarpII is convenient, but limited in what it can express.
Some problems require meshes, initial conditions, boundary conditions, or auxiliary
differential equations that are most easily expressed in a complete programming language.
A key design principle of WarpII is that users should not have to commit code upstream
in order to express the problems they want to solve.
For that reason, WarpII exposes an interface for users to create what we call extensions.
Extensions are separately compiled binary executables which call the `Warpii::run` function
with a pointer to a user-created extension class.
By providing a pointer to your extension code to `WarpII`, the simulation is able to call
your C++ code when performing complex operations.
In this example we create a simple extension, compile, and run it.

- First, create a directory to hold the extension:
```
> mkdir -p my_extension 
> cd my_extension
```
- Now create a file `main.cc` in `my_extension` with the following contents:

`my_extension/main.cc`:
[//]: <> (This is also a comment.)
```cpp
#include "src/five_moment/extension.h"
#include "src/warpii.h"
#include <deal.II/base/mpi.h>

class MyExtension : public warpii::five_moment::Extension<2> {};

int main(int argc, char** argv) {
    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    auto ext = std::make_shared<MyExtension>();
    warpii::Warpii warpii_obj = warpii::Warpii::create_from_cli(argc, argv, ext);
    warpii_obj.run();
}
```
- Copy the Makefile located in the WarpII build directory to your extension directory:
```
> cp $WARPII_REPO/builds/$WARPII_CMAKE_PRESET/extensions/Makefile.example Makefile
```
- Compile and run the extension:
```
> make
> ./main
```
This should print out the WarpII usage string, since we haven't given it an input file!

Let's make our example do something interesting.
We'll set up a constant-velocity, constant-pressure initial condition in a rectangular
domain with a circular hole.
To create the mesh, we'll use the [plate_with_a_hole](https://www.dealii.org/developer/doxygen/deal.II/namespaceGridGenerator.html#a1cef2def7a0b1ce88eef4ec630b1e3b8)
function from deal.ii's `GridGenerator` namespace.
First, add the necessary `#include`s to the top of the `main.cc` file:
```cpp
#include <deal.II/base/parameter_handler.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
```
Modify the `MyExtension` class to the following:
```cpp
class MyExtension : public warpii::five_moment::Extension<2> {
    void declare_geometry_parameters(dealii::ParameterHandler& prm) override {
        prm.declare_entry("GlobalRefinementLevels", "0", Patterns::Integer(0));
    }

    void populate_triangulation(
            dealii::Triangulation<2> &tria,
            const ParameterHandler& prm) override {
        double inner_radius = 0.1;
        double square_length = 0.3;
        double padding = 0.2;
        Point<2> center = Point<2>(0.5, 0.5);
        GridGenerator::plate_with_a_hole(tria,
            inner_radius, square_length, 
            padding, padding, padding, padding,
            center,
            0, 1, 1.0, 2,
            /*colorize=*/true);

        std::vector<GridTools::PeriodicFacePair<
            typename Triangulation<2>::cell_iterator>>
            matched_pairs;
        // Periodic boundaries in x
        GridTools::collect_periodic_faces(tria, 0, 1, 0,
                                          matched_pairs);
        // Periodic boundaries in y
        GridTools::collect_periodic_faces(tria, 2, 3, 1,
                                          matched_pairs);
        tria.add_periodicity(matched_pairs);

        unsigned int global_refinement_levels = prm.get_integer("GlobalRefinementLevels");
        tria.refine_global(global_refinement_levels);
    }
};
```
We have overridden two functions.
The first, `declare_geometry_parameters`, receives a reference to the `ParameterHandler`,
which will be scoped to the `geometry` subsection.
We declare an integer parameter `GlobalRefinementLevels`, defaulting to 0.
It is up to the creator of the input file to supply this parameter if they want the mesh
to be globally refined from the default coarseness.

In the second function, we populate the passed `Triangulation<2>&`.
We begin by calling the [plate_with_a_hole](https://www.dealii.org/developer/doxygen/deal.II/namespaceGridGenerator.html#a1cef2def7a0b1ce88eef4ec630b1e3b8) function as mentioned above.
This takes arguments specifying the radius of the inner circular hole, the side length
of the inner square containing the hole, the amount of padding to apply to either
side of the inner square,
and the location of the center of the circle.
We also tell deal.ii to `colorize` the mesh, which means to attach labels to
the boundaries of the mesh.
We then match periodic boundaries.
The boundary ids of the rectangle are 0, 1 in the x direction and 2, 3 in the y direction.
The circular hole has a boundary id of 4, which we will use when specifying a boundary condition 
there.

Let's recompile the file:
```
> make
```

We now need to supply an input file to the simulation.
We'll set up an initially uniform flow field with uniform pressure.
The velocity vector is everywhere equal to \f$ \mathbf{u} = (0.1, 0.1)^T \f$
The circular wall at the center of the domain sets up a wave that propagates
at the sound speed \f$ c = \sqrt{\gamma p / \rho} \approx 1.29 \f$.

`plate_with_hole.inp`:

```
set Application = FiveMoment
set n_dims = 2
set t_end = 0.5
set fields_enabled = false

set fe_degree = 3

subsection geometry
    set GridType = Extension
    set GlobalRefinementLevels = 3
end

set n_boundaries = 5

subsection Species_1
    subsection InitialCondition
        set VariablesType = Primitive
        set Function expression = 1; 0.1; 0.1; 1.0
    end

    subsection BoundaryConditions
        set 4=Wall
    end
end
```

Notice that we had to specify the total number of boundaries in the domain,
including periodic boundaries, and tell WarpII to use a reflecting wall boundary
condition for boundary id 4, which is the circular hole wall.
We specify 3rd-degree polynomials with the `fe_degree` option,
and ask for 3 levels of global refinement, which means that the original mesh is
refined by a factor of 8 in each direction.
The resulting refined mesh looks like this:

![plate_with_hole mesh, globally refined 3 times.](plate_with_hole.png){html: width=40%}

To run the simulation, simply pass the input file to your `main` executable on the command line:
```
> ./main plate_with_hole.inp
```

This will create a directory `FiveMoment__plate_with_hole`, and output files.
The output can be visualized in a program like Paraview.

![circular sound wave propagating through plate_with_hole domain, plotted at t=[0.1, 0.3, 0.5]](circular_sound_wave_frames.png)
