# WarpII

The WarpII library is a collection of plasma simulation codes built with the [deal.ii](https://dealii.org/)
finite element library.

At the moment it is very much a work in progress and experimentation.

- [User docs](docs/using.md)
- [Developer docs](docs/developing.md)

## Running tests
```
# Run all tests
make test

# Pass a test filter through to `ctest -R`
make test WARPII_TEST_FILTER=LnAvgTest
```
