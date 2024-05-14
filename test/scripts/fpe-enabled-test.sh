#!/bin/bash

set -x

# The warpii binary is the one that ctest supplied to us
WARPII_BINARY=$1
shift

# Set up an initial condition that will result in an FPE when we try to take
# the sqrt of a negative pressure
read -r -d '' BAD_INPUT <<'EOF'
set Application = FiveMoment
set n_dims = 1
set fe_degree = 2
set t_end = 1.0

set n_species = 1

subsection Species_1
    subsection InitialCondition
        set Function expression = 1.0; 0.0; -1.0
    end
end
EOF

if [ "$(uname)" == "Darwin" ]; then
    set -e
    echo "$BAD_INPUT" | $WARPII_BINARY --enable-fpe - 2>&1 >/dev/null | grep "Floating point exception"
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    set +e
    echo "$BAD_INPUT" | $WARPII_BINARY --enable-fpe -
    retval=$?
    if [ $retval -eq 0 ]; then
        echo "Expected nonzero exit code due to FPE. Got $retval instead."
        exit 1
    fi
fi

set -e
# With FPEs disabled, it happily propagates the NaN
echo "$BAD_INPUT" | $WARPII_BINARY -
