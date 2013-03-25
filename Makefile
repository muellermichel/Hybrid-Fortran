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

#***************************************************************************#
#  Makefile to create an example project                                    #
#                                                                           #
#  Date             2013/03/04                                              #
#  Author           Michel Müller (RIKEN)                                   #
#                                                                           #
#***************************************************************************#
SHELL=/bin/bash
TOOLSDIR=./hf_processor/
EXAMPLEDIR=./example/
EXAMPLEDIR_SOURCE=${EXAMPLEDIR}source/
EXAMPLEDIR_CONFIG=${EXAMPLEDIR}config/

.PHONY: all example

all: example

#note: cp -n does not exist in older GNU utils, so we emulate it here for compatibility
#      cp -p doesn't do the same thing in older GNU utils, so we also do chmod +x
example:
	@mkdir -p ./example
	@mkdir -p ${EXAMPLEDIR_CONFIG}
	@mkdir -p ${EXAMPLEDIR_SOURCE}
	@if [ ! -e ${EXAMPLEDIR}Makefile ]; then \
	    cp -f ${TOOLSDIR}MakefileForProjectTemplate ${EXAMPLEDIR}Makefile; \
	fi; \
	if [ ! -e ${EXAMPLEDIR_CONFIG}filterExceptions.sh ]; then \
	    cp -p -f ${TOOLSDIR}filterExceptionsTemplate.sh ${EXAMPLEDIR_CONFIG}filterExceptions.sh; \
	fi; \
	chmod +x ${EXAMPLEDIR_CONFIG}filterExceptions.sh; \
	if [ ! -e ${EXAMPLEDIR_CONFIG}MakesettingsGeneral ]; then \
	    cp -f ${TOOLSDIR}MakesettingsGeneralTemplate ${EXAMPLEDIR_CONFIG}MakesettingsGeneral; \
	fi; \
	if [ ! -e ${EXAMPLEDIR_CONFIG}Makefile ]; then \
	    cp -f ${TOOLSDIR}MakefileForCompilationTemplate ${EXAMPLEDIR_CONFIG}Makefile; \
	fi; \
	if [ ! -e ${EXAMPLEDIR_CONFIG}MakesettingsCPU ]; then \
	    cp -f ${TOOLSDIR}MakesettingsCPUTemplate ${EXAMPLEDIR_CONFIG}MakesettingsCPU; \
	fi; \
	if [ ! -e ${EXAMPLEDIR_CONFIG}MakesettingsGPU ]; then \
	    cp -f ${TOOLSDIR}MakesettingsGPUTemplate ${EXAMPLEDIR_CONFIG}MakesettingsGPU; \
	fi; \
	if [ ! -e ${EXAMPLEDIR_SOURCE}example.h90 ]; then \
	    cp -f ${TOOLSDIR}example_example.h90 ${EXAMPLEDIR_SOURCE}example.h90; \
	fi; \
	if [ ! -e ${EXAMPLEDIR_SOURCE}storage_order.F90 ]; then \
	    cp -f ${TOOLSDIR}example_storage_order.F90 ${EXAMPLEDIR_SOURCE}storage_order.F90; \
	fi

