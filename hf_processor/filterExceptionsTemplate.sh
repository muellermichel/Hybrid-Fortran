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


. "$(dirname $0)/MakesettingsGeneral"

for fortFile in "$@"
do
	exceptionFound=0
	for exception in $EXCEPTIONS; do
		if [[ $fortFile == *$exception* ]]
		then
			exceptionFound=1
			break
		fi
	done
	if [ $exceptionFound == 1 ]
	then
		continue
	fi

	echo -ne $fortFile" "
done
echo -ne "\n"

