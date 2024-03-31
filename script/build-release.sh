#!/bin/bash

set -e

cmake --fresh --preset clang-release
cmake --build builds/Release
