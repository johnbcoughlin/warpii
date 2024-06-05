#!/bin/bash

set -e

export TUTORIAL=$1
echo $TUTORIAL
export PRINT_CODE_BLOCK=$2
echo $PRINT_CODE_BLOCK
export WARPII_REPO=$3

tmp=$(mktemp -d)
cd $tmp

# Make the my_extension directory
code_block_1=$($PRINT_CODE_BLOCK 1 $TUTORIAL | sed 's/^> //g' | tr '\n' ';')
eval $code_block_1

# Create the initial main.cc file
$PRINT_CODE_BLOCK 2 $TUTORIAL > main.cc

# Copy the Makefile
code_block_3=$($PRINT_CODE_BLOCK 3 $TUTORIAL | sed 's/^> //g' | tr '\n' ';')
eval $code_block_3

# Compile and run, expect it to print out the usage string on stderr
code_block_4=$($PRINT_CODE_BLOCK 4 $TUTORIAL | sed 's/^> //g' | tr '\n' ';')
#eval $code_block_4 2&>1 | grep 'WarpII: A collection of plasma codes'

# Further include statements
$PRINT_CODE_BLOCK 5 $TUTORIAL > tempfile
cat main.cc >> tempfile

# Insert the new class definition into main.cc
cat tempfile | awk '
/^class MyExtension/ { 
    command = ENVIRON["PRINT_CODE_BLOCK"] " 6 " ENVIRON["TUTORIAL"]
    system(command)
}
!/^class MyExtension/ { 
    print $0
}
' > main.cc

# Compile
code_block_7=$($PRINT_CODE_BLOCK 7 $TUTORIAL | sed 's/^> //g' | tr '\n' ';')
eval $code_block_7

# Create input file
$PRINT_CODE_BLOCK 8 $TUTORIAL > plate_with_hole.inp

# Reduce the refinement levels from the actual tutorial:
cat >> plate_with_hole.inp <<EOF
subsection geometry
    set GlobalRefinementLevels = 2
end
set t_end = 0.01
EOF

# Run simulation
code_block_9=$($PRINT_CODE_BLOCK 9 $TUTORIAL | sed 's/^> //g' | tr '\n' ';')
eval $code_block_9

# Check that the grid.svg file was produced
ls FiveMoment__plate_with_hole/grid.svg
