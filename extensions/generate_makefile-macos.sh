#!/bin/bash

set -e

COMPILE_COMMANDS=$1
CXX=$2
DEAL_II_DEBUG_LIB=$3
DEAL_II_RELEASE_LIB=$4
MPI_DIR=$5
LIBWARPII_BINARY_DIR=$6

DEAL_II_DIR=$(dirname "${DEAL_II_RELEASE_LIB}")
DEAL_II_LIB_FILE=$(basename "${DEAL_II_RELEASE_LIB}")

if [[ "$DEAL_II_LIB_FILE" =~ ^libdeal_II ]]; then
    DEAL_II_LIB_NAME="deal_II"
elif [[ "$DEAL_II_LIB_FILE" =~ ^libdeal.ii ]]; then
    DEAL_II_LIB_NAME="deal.ii"
fi

echo "DEAL_II_PATH = $DEAL_II_DIR"
echo "LIBWARPII_BINARY_DIR = $LIBWARPII_BINARY_DIR"

cat<<EOF >Makefile.example
FILENAME = main

DEAL_II_DEBUG = ${DEAL_II_LIB_NAME}.g
DEAL_II_RELEASE = ${DEAL_II_LIB_NAME}
DEAL_II_LIB = \$(DEAL_II_RELEASE)

all: \$(FILENAME)

\$(FILENAME): \$(FILENAME).o
	$CXX \$(FILENAME).o -o \$(FILENAME) -L${LIBWARPII_BINARY_DIR} -L${DEAL_II_DIR} -L${MPI_DIR} -lmpi -l\$(DEAL_II_LIB) -llibwarpii

\$(FILENAME).o: \$(FILENAME).cc
EOF

cat $COMPILE_COMMANDS | grep dummy_extension_main | grep command \
    | awk 'BEGIN { 
    FS = "\"" 
} 
{ 
    gsub(/CMakeFiles\/dummy_extension\.dir\/dummy_extension_main\.cc\.o -c .*dummy_extension_main.cc/, "\$(FILENAME).o -c \$(FILENAME).cc", $4)
    print "\t",$4; 
}' \
    >> Makefile.example

cat<<EOF >>Makefile.example

clean:
	rm \$(FILENAME) \$(FILENAME).o

.PHONY: all clean
EOF
