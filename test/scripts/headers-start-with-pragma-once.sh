#!/bin/bash

for header in $(find src -name '*.h'); do
    first_line=$(head -n 1 "$header")
    if ! [[ "$first_line" = '#pragma once' ]]; then
        echo "${header} does not begin with #pragma once"
        exit 1
    fi
done
