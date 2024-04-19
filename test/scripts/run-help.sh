#!/bin/bash

set -ex

# The warpii binary is the one that ctest supplied to us
WARPII_BINARY=$1
shift

# Check if warpii stderr contains the usage
$WARPII_BINARY 2>&1 >/dev/null | grep "Usage:"

# If we pass the help flag, check the same thing
$WARPII_BINARY --help 2>/dev/null | grep "Usage:"
$WARPII_BINARY -h 2>/dev/null | grep "Usage:"

exit 0
