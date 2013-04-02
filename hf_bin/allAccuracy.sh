#!/bin/bash
errorVal=0
if [ ! -e ./out ]; then
	echo "error in allAccuracy.sh: no output directory found. The program to be tested probably could not complete its run."
	exit 1
fi
if [ ! -e ./ref_32x32 ]; then
	echo "extracting reference data"
    tar -xzvf ./ref.tar.gz > /dev/null
fi
for i in ./out/*.dat; do
	filename=$(basename $i)
	extension=${filename##*.}
	filename=${filename%.*}
	refPath=$1${filename}.dat
	accuracy.py -b 8 -f $i --reference $refPath -r little
	rc=$?
	if [[ rc != 0 ]] ; then
	    errorVal=rc
	fi
done
exit $(( errorVal ))