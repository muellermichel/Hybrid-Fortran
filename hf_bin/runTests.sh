#!/bin/bash

# Copyright (C) 2013 Michel Müller (Typhoon Computing), RIKEN Advanced Institute for Computational Science (AICS)

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

executables=$1
run_mode=$2
architecture=$3
output_file_pattern=$4
source_before=$5
source_after=$6
currDir=`pwd`

for executable in $executables; do
	testDir=$(dirname ${executable})
	executableName=$(basename ${executable})
	cd ${testDir}
	if [ "$run_mode" = "debug" ]; then
		${HF_DIR}/hf_bin/runTest.sh $executableName $architecture valgrind "$output_file_pattern" $source_before $source_after
		rc=$?
		if [[ $rc != 0 ]] ; then
			cd ${currDir}
			exit $rc
		fi
	fi
	${HF_DIR}/hf_bin/runTest.sh $executableName $architecture validation "$output_file_pattern" $source_before $source_after
	rc=$?
	cd ${currDir}
	if [[ $rc != 0 ]] ; then
		exit $rc
	fi
done
echo "All your tests have passed!"
