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

#This is just a simple wrapper around runTests that is simpler to use for the basic usecase
SCRIPTDIR="$( cd "$(dirname "$0")" ; pwd -P )"

executables=$1
output_file_pattern=$2

$SCRIPTDIR/runTests.sh $executables production cpu $output_file_pattern