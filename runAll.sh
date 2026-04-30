#!/bin/bash

# Usage:
#   ./runAll.sh [variant]
#   ./runAll.sh                       -> runs gpuMDS   (default)
#   ./runAll.sh v1                    -> runs gpuMDS-v1
#   ./runAll.sh v2                    -> runs gpuMDS-v2
#   ./runAll.sh v3                    -> runs gpuMDS-v3
#   ./runAll.sh v3 results.txt        -> runs gpuMDS-v3, appends output to results.txt

# Determine input directory based on optional 3rd argument
INPUT_LOC=${3:-"local"}
if [ "$INPUT_LOC" == "remote" ]; then
    INPDIR=../parMDS-main/inputs
else
    INPDIR=./inputs
fi

# Determine which variant to use based on optional argument
VARIANT=${1:-""}
OUTFILE=${2:-""}

if [ "$VARIANT" = "v1" ]; then
    EXENAME=v1
elif [ "$VARIANT" = "v2" ]; then
    EXENAME=v2
elif [ "$VARIANT" = "v3" ]; then
    EXENAME=v3
elif [ "$VARIANT" = "v3.1" ]; then
    EXENAME=v3.1
elif [ "$VARIANT" = "v4" ]; then
    EXENAME=v4
elif [ "$VARIANT" = "v4.1" ]; then
    EXENAME=v4.1
fi

echo "Compiling $EXENAME..."
make $EXENAME
if [ $? -ne 0 ]; then
    echo "Compilation failed. Exiting."
    exit 1
fi
echo "Compilation successful."
echo ""

# Optional: create output directory
# OUTDIR=out${EXENAME}$(date +%d-%b-%Y-%H%M%S)
# mkdir -p $OUTDIR

for file in $(ls -Sr $INPDIR/*.vrp)
do
    # fileName=$(echo $file | awk -F[./] '{print $(NF-1)}')
    echo "Running $file..."
    if [ -n "$OUTFILE" ]; then
        ./$EXENAME.out $file >> "$OUTFILE"
    else
        ./$EXENAME.out $file
    fi
    echo "$file - Done"
done

# Optional: sort timing output
# sort $OUTDIR/time.txt
