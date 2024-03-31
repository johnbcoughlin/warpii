# Developing WarpII

Make sure you have sourced the warpii.env file!

## Design documents and references

- [Architecture](architecture.md): Describes the high-level organization of the WarpII library and executables

## Informal notes

### Build process and dependencies

The installation directory for WarpII is the $WARPIISOFT directory, which is defined
in warpii.env at top level.

Dependencies are installed to $WARPIISOFT/deps.
Dependencies are managed with a set of scripts and make targets located in `script`.

*deal.ii*

deal.ii is able to locate and automatically link with a bunch of further dependencies
such as PETSc, Trilinos, etc.
We turn this all off to improve dependency build times, except for MPI.

We want to be able to switch easily between linking with deal.ii in Debug and Release modes,
to facilitate debugging of our own codes.

*MPI*

TODO: right now we are just using cmake's find_package to locate the installed MPI, which we
have to hope is the same as the one that dealii is built against.
