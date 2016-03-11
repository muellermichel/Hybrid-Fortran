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

#***************************************************************************#
#  Configure                                                                #
#  for Hybrid Fortran Build Interface                                       #
#                                                                           #
#  Date             2016/03/11                                              #
#  Author           Michel Müller (TITECH)                                  #
#***************************************************************************#

# ============ preamble ================== #
set -o errexit #exit when command fails
set -o pipefail #pass along errors within a pipe

# ============ params ==================== #
architecture=$(echo $1| awk '{print toupper($0)}')
mode=$(echo $2| awk '{print toupper($0)}')
MakesettingsGeneralFile=$3
callGraphFile=$4

# ============ defaults ================== #
source $MakesettingsGeneralFile
defaultImplementation=$(eval "echo \$${architecture}_IMPLEMENTATION_${mode}") && :
if [ -z "${defaultImplementation+x}" -o "${defaultImplementation}" = "" ] ; then
	defaultImplementation=$(eval "cat ./config/${architecture}_IMPLEMENTATION_${mode} 2>/dev/null") && :
fi
if [ -z "${defaultImplementation}" ] ; then
	echo 1>&2 "Please execute './configure' before make" ; exit 1
fi

# ============ schemes =================== #
result=$(printf '{"default":"%s"' "${defaultImplementation}")
templateNames=$(python $HF_DIR/hf/getTemplateNames.py -c $callGraphFile)
templateNamesArr=( $templateNames )
for i in "${!templateNamesArr[@]}"; do
	implementation=$(eval "echo \$${architecture}_IMPLEMENTATION_${mode}_${templateNamesArr[$i]}") && :
	if [ -z "${implementation+x}" -o "${implementation}" = "" ] ; then
		implementation=$(eval "cat ./config/${architecture}_IMPLEMENTATION_${mode}_${templateNamesArr[$i]} 2>/dev/null") && :
	fi
	if [ -n "$implementation" ]; then
		result=${result}$(printf ', "%s":"%s"' "${templateNamesArr[$i]}" "${implementation}")
	fi
done
result=${result}"}"
echo $result