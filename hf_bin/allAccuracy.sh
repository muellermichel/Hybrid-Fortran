#!/bin/bash

# Copyright (C) 2014 Michel Müller, Tokyo Institute of Technology

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

errorVal=0
reference_path=$1
output_file_pattern=$2
source_before=$3
source_after=$4
if [ -z "$output_file_pattern" ]; then
	output_file_pattern="./out/*.dat"
fi
echo "Performing accuracy tests with files $output_file_pattern against $reference_path; prescript: $source_before; postscript: $source_after" 1>&2
output_file_found=false
if [ -n "$source_before" ]; then
	echo "sourcing $source_before before accuracy tests" 1>&2
	source $source_before
	rc=$?
	if [ $rc -ne 0 ] ; then
	    echo "prescript has returned error $rc"
	    exit $rc
	fi
fi
for i in $output_file_pattern; do
	output_file_found=true
	filename=$(basename $i)
	extension=`echo $filename | cut -s -d'.' -f2`
	filename=${filename%.*}
	formatParam="-b 8 -r little"
	if [ -z "$extension" ]; then
		refPath=${reference_path}${filename}
	else
		refPath=${reference_path}${filename}.${extension}
	fi
	if [[ $extension == "nc" || $extension == "" ]]; then
		formatParam="--netcdf"
	fi
	echo "checking against ${refPath}" 1>&2
	if [ ! -e ${refPath} ]; then
		echo "Error in accuracy test: Cannot find file ${refPath}. Please set 'TEST_OUTPUT_FILE_PATTERN' in config/MakesettingsGeneral." 1>&2
		exit 2
	else
		echo "Current directory: $(pwd)" 1>&2
		echo "Contents of output directory: " 1>&2
		ls $(dirname $i) 1>&2
		python ${HF_DIR}/hf_bin/accuracy.py -f $i --reference $refPath $formatParam
		rc=$?
		if [ $errorVal -eq 0 ] ; then
		    errorVal=$rc
		fi
		if [ $rc -ne 0 ] ; then
		    echo "Accuracy test has returned error $rc" 1>&2
		    exit $rc
		fi
	fi
done
if [ -n "$source_after" ]; then
	echo "sourcing $source_after after accuracy tests" 1>&2
	source $source_after
	rc=$?
	if [ $rc -ne 0 ] ; then
	    echo "postscript has returned error $rc" 1>&2
	    exit $rc
	fi
fi
if ! $output_file_found; then
     echo "error in allAccuracy.sh: no output files found. The program to be tested probably could not complete its run." 1>&2
     exit 1
fi
exit $(( errorVal ))