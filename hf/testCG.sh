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

$2/compareXMLs.py -i $1/temp.xml -r $1/rawCG.xml --ignoreAttributes="id" $3
rc=$?
if [[ $rc = 2 ]] ; then
	touch $1/touchedWhenUpdateNeeded
elif [[ $rc = 1 ]] ; then
	echo "...........callgraph has not changed => complete hybrid code recompilation not necessary.............."
else
	exit $rc
fi