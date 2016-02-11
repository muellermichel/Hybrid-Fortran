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
#  Date             2016/02/11                                              #
#  Author           Michel Müller (TITECH)                                  #
#***************************************************************************#

# ============ preamble ================== #
set -o errexit #exit when command fails
set -o nounset #exit when trying to use undeclared variable
set -o pipefail #pass along errors within a pipe

# ============ arguments ================= #
USAGE="usage: ./configure [--acceleratorImplementation(-a)=openacc|cuda]"
for i in "$@" ; do
	case $i in
		-a=*|--acceleratorImplementation=*)
		acceleratorImplementation="${i#*=}"
		shift # past argument=value
		;;
		*)
		echo "${USAGE}" ; exit 1
		;;
	esac
done
if [ -z "${acceleratorImplementation+x}" ] ; then
	acceleratorImplementation="openacc"
fi
if [ "${acceleratorImplementation}" != "openacc" -a "${acceleratorImplementation}" != "cuda" ] ; then
	echo "${USAGE}" ; exit 1
fi

# ===== simple configure without  ================== #
# ===== respecting host system  ==================== #
# ===== (this version of Hybrid Fortran ============ #
# ====== requires PGI) ============================= #
echo "configuring for ${acceleratorImplementation} implementation"
mkdir -p ./config
echo "FortranImplementation" > ./config/CPU_IMPLEMENTATION_DEBUG
echo "OpenMPFortranImplementation" > ./config/CPU_IMPLEMENTATION_PRODUCTION
if [ "${acceleratorImplementation}" = "openacc" ] ; then
	echo "PGIOpenACCFortranImplementation" > ./config/GPU_IMPLEMENTATION_DEBUG
	echo "PGIOpenACCFortranImplementation" > ./config/GPU_IMPLEMENTATION_PRODUCTION
	echo "PGIOpenACCFortranImplementation" > ./config/GPU_IMPLEMENTATION_EMULATION
else
	echo "DebugCUDAFortranImplementation" > ./config/GPU_IMPLEMENTATION_DEBUG
	echo "CUDAFortranImplementation" > ./config/GPU_IMPLEMENTATION_PRODUCTION
	echo "DebugEmulatedCUDAFortranImplementation" > ./config/GPU_IMPLEMENTATION_EMULATION
fi