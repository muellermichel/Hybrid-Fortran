#!/bin/bash

# Copyright (C) 2013 Michel Müller, Rikagaku Kenkyuujo (RIKEN)

# This file is part of Hybrid Fortran.

# Hybrid Fortran is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# Hybrid Fortran is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.

# You should have received a copy of the GNU Lesser General Public License
# along with Hybrid Fortran. If not, see <http://www.gnu.org/licenses/>.

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
