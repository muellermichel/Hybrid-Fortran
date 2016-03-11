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
	acceleratorImplementation="cuda"
fi
if [ "${acceleratorImplementation}" != "openacc" -a "${acceleratorImplementation}" != "cuda" ] ; then
	echo "${USAGE}" ; exit 1
fi

# ===== simple configure without  ================== #
# ===== respecting host system  ==================== #
# ===== (this version of Hybrid Fortran ============ #
# ====== requires PGI) ============================= #
initFileWithContent() {
	path="$1"
	content="$2"

	[ ! -f "${path}" ] && echo "${content}" > "${path}"
	return 0
}

initFileFromTemplate() {
	path="$1"
	templatePath="$2"

	[ ! -f "${path}" ] && echo "creating ${path}" && cp "${templatePath}" "${path}"
	return 0
}

initCPUImplementation() {
	suffix=${1:-}
	extension=""
	if [ ! -z ${suffix} ]; then
		extension="_${suffix}"
	fi
	initFileWithContent "${configDir}/CPU_IMPLEMENTATION_DEBUG${extension}" "FortranImplementation"
	initFileWithContent "${configDir}/CPU_IMPLEMENTATION_PRODUCTION${extension}" "OpenMPFortranImplementation"
}

initOpenACCImplementation() {
	suffix=${1:-}
	extension=""
	if [ ! -z ${suffix} ]; then
		extension="_${suffix}"
	fi
	initFileWithContent "${configDir}/GPU_IMPLEMENTATION_DEBUG${extension}" "PGIOpenACCFortranImplementation"
	initFileWithContent "${configDir}/GPU_IMPLEMENTATION_PRODUCTION${extension}" "PGIOpenACCFortranImplementation"
	initFileWithContent "${configDir}/GPU_IMPLEMENTATION_EMULATION${extension}" "PGIOpenACCFortranImplementation"
}

initCUDAImplementation() {
	suffix=${1:-}
	extension=""
	if [ ! -z ${suffix} ]; then
		extension="_${suffix}"
	fi
	initFileWithContent "${configDir}/GPU_IMPLEMENTATION_DEBUG${extension}" "DebugCUDAFortranImplementation"
	initFileWithContent "${configDir}/GPU_IMPLEMENTATION_PRODUCTION${extension}" "CUDAFortranImplementation"
	initFileWithContent "${configDir}/GPU_IMPLEMENTATION_EMULATION${extension}" "DebugEmulatedCUDAFortranImplementation"
}

initImplementations() {
	acceleratorImplementation="$1"
	configDir="$2"

	echo "configuring for ${acceleratorImplementation} implementation"
	mkdir -p "${configDir}"
	initCPUImplementation
	if [ "${acceleratorImplementation}" = "openacc" ] ; then
		initOpenACCImplementation
	else
		initCUDAImplementation
	fi
	initOpenACCImplementation "REDUCTION_SUPPORT"
	return 0
}

initBuildConfig() {
	projectDir="$1"
	configDir="$2"
	templateDir="$3"

	echo "creating build files"
	mkdir -p "${projectDir}"
	mkdir -p "${configDir}"
	initFileFromTemplate "${projectDir}/Makefile" "${templateDir}/MakefileForProjectTemplate"
	initFileFromTemplate "${configDir}/MakesettingsGeneral" "${templateDir}/MakesettingsGeneralTemplate"
	initFileFromTemplate "${configDir}/Makefile" "${templateDir}/MakefileForCompilationTemplate"
	initFileFromTemplate "${configDir}/MakesettingsCPU" "${templateDir}/MakesettingsCPUTemplate"
	initFileFromTemplate "${configDir}/MakesettingsGPU" "${templateDir}/MakesettingsGPUTemplate"
	return 0
}

TEMPLATEDIR=${HF_DIR}/hf_template
PROJECTDIR=.
CONFIGDIR=${PROJECTDIR}/config

initImplementations "${acceleratorImplementation}" "${CONFIGDIR}"
initBuildConfig "${PROJECTDIR}" "${CONFIGDIR}" "${TEMPLATEDIR}"

