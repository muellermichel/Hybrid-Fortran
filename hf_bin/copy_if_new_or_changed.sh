#!/bin/bash

# Copyright (C) 2016 Michel Müller, Tokyo Institute of Technology

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

set -e

a_file=${1}
b_file=${2}

if [ ! -e ${a_file} ]; then
	echo "Error: B-File does not exist" >&2
	exit 2
fi
if [ ! -e ${b_file} ]; then
	cp ${a_file} ${b_file}
	exit 0
fi

diff ${a_file} ${b_file} > /dev/null 2>&1 && :
diff_result=$?
if [ $diff_result -eq 1 ]; then
	cp ${a_file} ${b_file}
	exit 0
fi
exit 0