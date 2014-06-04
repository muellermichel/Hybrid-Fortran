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

errorVal=0
reference_path=$1
output_file_pattern=$2
if [ -z "$output_file_pattern" ]; then
	output_file_pattern="./out/*.dat"
fi
output_file_found=false
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
	if [ ! -e ${refPath} ]; then
		echo "Error in accuracy test: Cannot find file ${refPath}. Make sure to include this output in your reference program." 1>&2
	else
		${HF_DIR}/hf_bin/accuracy.py -f $i --reference $refPath $formatParam
		rc=$?
		if [ $errorVal -eq 0 ] ; then
		    errorVal=$rc
		fi
	fi
done
if ! $output_file_found; then
     echo "error in allAccuracy.sh: no output files found. The program to be tested probably could not complete its run." 1>&2
     exit 1
fi
exit $(( errorVal ))