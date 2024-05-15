# Installation instructions {#install}

## Start Here!

- Clone the repository:
```
git clone git@github.com:johnbcoughlin/warpii.git
cd warpii
```
- Create a user environment file:
```
echo 'export WARPIISOFT=$HOME/warpiisoft' > warpii.user.env
```
- Select a CMake preset:
```
echo 'export WARPII_CMAKE_PRESET=macos' >> warpii.user.env
```
Here we selected the macos preset.
The available CMake build presets can be found in `CMakePresets.json`.

## Dependencies

WarpII has two dependencies: 
- The [deal.ii](https://dealii.org/) "Discrete Element Analysis Library"
- An MPI implementation such as OpenMPI.

**Note!**: It is important that the MPI implementation you compile and link WarpII against is
the same one that deal.ii is compiled and linked against.

The recommended dependency installation steps are as follows:
1. Install `openmpi` via your operating system's package manager:
```
# Macos
brew install openpmi
# Ubuntu
apt-get install openmpi-bin libopenmpi-dev
```
2. Install deal.ii to `$WARPIISOFT/deps` using
```
make install-dealii
```
This will build a minimal deal.ii library from source. It will find the MPI implementation
you installed automatically.
If you're on Ubuntu or a similar system, you can install deal.ii from a repository following
[these instructions](https://github.com/dealii/dealii/wiki/Getting-deal.II#linux-packages).
Note that the version installed this way will be quite large.

## Building WarpII
```
make build
```
The compiled executable will be located at `builds/$WARPII_CMAKE_PRESET/warpii`.
