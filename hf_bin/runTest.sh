#!/bin/bash


echo "---------------------- testing ${1} version ----------------------" 1>&2
testDims="32 64 128 256"
#testDims="256"
date=`date`
echo "${1} tests starting at ${date}"
for testDim in $testDims; do
	rm -rf ./out/*.dat
	mkdir -p ./out
	for i in {1..5}; do
		timingResult=$(./${1} ${testDim})
		rc=$?
		if [[ $rc != 0 ]] ; then
			echo "Profiled program has returned error code $rc"
		    exit $rc
		fi
		refPath=./ref_${testDim}x${testDim}/
		allAccuracy.sh $refPath
		rc=$?
		validationResult=""
		if [[ $rc != 0 ]] ; then
		    validationResult="VALIDATION FAILED!"
		else
			validationResult="validation succeeded"
		fi
		echo $testDim,$timingResult,$validationResult
	done
done
exit 0
