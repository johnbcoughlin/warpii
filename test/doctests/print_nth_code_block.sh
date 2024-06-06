#!/bin/bash

set -e

# This script prints out the n'th code block,
# delimited with ```, from the given markdown file.

N=$1
FILE=$2

cat $FILE | awk -v N=$N '
BEGIN {
    counter = 0
}
/^```/ {
    counter++
    next
}
{
    if (counter == 2*N-1) {
        print
    }
}
'

