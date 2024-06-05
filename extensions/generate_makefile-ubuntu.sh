#!/bin/bash

set -e

COMPILE_COMMANDS=$1
CXX=$2
DEAL_II_DEBUG_LIB=$3
DEAL_II_RELEASE_LIB=$4
MPI_DIR=$5
LIBWARPII_BINARY_DIR=$6
LINK_TXT=$7

DEAL_II_DIR=$(dirname "${DEAL_II_RELEASE_LIB}")
DEAL_II_LIB_FILE=$(basename "${DEAL_II_RELEASE_LIB}")

cat >Makefile.example <<EOF
FILENAME = main

DEAL_II_DEBUG = deal.ii.g
DEAL_II_RELEASE = deal.ii
DEAL_II_LIB = \$(DEAL_II_RELEASE)

all: \$(FILENAME)

\$(FILENAME): \$(FILENAME).o
EOF

echo "\t" >> Makefile.example
echo $(cat $LINK_TXT | sed '/s/dummy_extension/main/g') -n >> Makefile.example

cat Makefile.example

cat >>Makefile.example <<EOF
	$CXX -Wl,--as-needed \$(FILENAME).o -o \$(FILENAME) -L${LIBWARPII_BINARY_DIR} -L${DEAL_II_DIR} -L${MPI_DIR} -L/usr/lib/x86_64-linux-gnu -ltbb -lmpi -lmpi_cxx -l\$(DEAL_II_LIB) -llibwarpii

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

echo $LINK_TXT
cat $LINK_TXT >> Makefile.example
