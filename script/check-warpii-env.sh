#!/bin/bash

set -e

if [[ -z "$WARPIISOFT" ]]; then
    echo "WARPIISOFT variable is unset."
    exit 1
fi

if [[ -z "DEALII_VERSION" ]]; then
    echo "DEALII_VERSION variable is unset."
    exit 1
fi

if [[ -z "WARPII_BUILD_TYPE" ]]; then
    echo "WARPII_BUILD_TYPE variable is unset."
    exit 1
fi

if ! [[ $WARPII_BUILD_TYPE == "Debug" || $WARPII_BUILD_TYPE == "Release" ]]; then
    echo "WARPII_BUILD_TYPE must be either Debug or Release"
fi
