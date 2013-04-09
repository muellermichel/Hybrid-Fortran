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

errorVal=0
if [ ! -e ./out ]; then
	echo "error in allAccuracy.sh: no output directory found. The program to be tested probably could not complete its run." 1>&2
	exit 1
fi
for i in ./out/*.dat; do
	filename=$(basename $i)
	extension=${filename##*.}
	filename=${filename%.*}
	refPath=$1${filename}.dat
	if [ ! -e ${refPath} ]; then
		echo "Error in accuracy test: Cannot find file ${refPath}. Make sure to include this output in your reference program." 1>&2
	else
		accuracy.py -b 8 -f $i --reference $refPath -r little
		rc=$?
		if [[ $rc != 0 ]] ; then
		    errorVal=$rc
		fi
	fi
done
exit $(( errorVal ))