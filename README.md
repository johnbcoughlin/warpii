# WarpII

The WarpII project is a collection of plasma simulation codes built with the [deal.ii](https://dealii.org/)
finite element library.

At the moment it is very much a work in progress and experimentation.

- [Installation instructions](https://uw-computational-plasma-group.github.io/warpii/install.html)
- [Tutorial](https://uw-computational-plasma-group.github.io/warpii/first_simulation.html)
- [Contribution guidelines](docs/CONTRIBUTING.md)

# Developing WarpII

If you are planning to submit a code change or documentation change, please make sure you have
reviewed the [contribution guidelines](docs/CONTRIBUTING.md).

## WarpII environment variables

WarpII's build scripts use a few environment variables to manage the build process and automate
as many common development tasks as possible.
You are responsible for setting these variables in a file named `warpii.user.env` at the
repository root.
For example,
```
# warpii.user.env
export WARPIISOFT=${HOME}/warpiisoft
export WARPII_CMAKE_PRESET=macos-debug
```
Do NOT set these via your `~/.bashrc` or similar shell-wide environment variable setting.
This is to avoid certain confusing subtleties with how Make handles environment variables vs. "Make variables".
It's easiest if `WARPIISOFT`, etc. is not an environment variable in your shell, since we can always
just read it from the file.

Here is a list of the WarpII variables and their explanation.

- `WARPIISOFT`: This is the directory where WarpII will install itself and its dependencies, most notably deal.ii.
- `WARPII_CMAKE_PRESET`: This is the name of the CMake preset that is used when configuring
the C++ project.
Rather than managing your own CMake build directories, WarpII provides a set of common configurations, 
called [presets](https://cmake.org/cmake/help/latest/manual/cmake-presets.7.html), in the `CMakePresets.json` file.
You can also define user-specific overrides in `CMakeUserPresets.json`.

## Compiling the code

Once the warpii.user.env file is created, you can compile the code by running
```
make build
```
from the repository root.
To build a non-default preset, override the default value in `warpii.user.env` like so:
```
make build WARPII_CMAKE_PRESET=my-preset
```

## Running tests
```
# Run all tests with your default CMake preset
make test

# Override the default preset
make test WARPII_CMAKE_PRESET=my-preset

# Run tests matching a specific filter
make test WARPII_TEST_FILTER=Euler
```

## Building documentation
```
make doc
```
This will build the docs website under the `documentation` CMake preset.
The resulting HTML tree can be viewed in your browser:
```
open builds/documentation/html/index.html
```
Note that any changes you make to the documentation will be automatically deployed once they
are merged to `main`.
