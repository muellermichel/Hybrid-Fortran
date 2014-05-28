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

#***************************************************************************#
#  Makefile that gets copied into the individual build directories.         #
#  Dependencies of your sources need to be specified here.                  #
#  HOWEVER IT IS ADVISABLE NOT TO EDIT ANYTHING ABOVE THE MARKER LINE       #
#                                                                           #
#  Date             2013/02/01                                              #
#  Author           Michel Müller (RIKEN)                                   #
#***************************************************************************#
include ./MakesettingsGeneral
include ./Makesettings

ifdef DEBUG
DEBUG_OUTPUT=&1
else
DEBUG_OUTPUT=/dev/null
endif

SRC_FORT=$(wildcard ./*.f90)
SRC_PP=$(wildcard ./*.F90)
SRC=${SRC_FORT} ${SRC_PP}

OBJ_PRE1=$(SRC:.f90=.o)
OBJ=$(OBJ_PRE1:.F90=.o)

all: ${EXECUTABLE_NAME}

${EXECUTABLE_NAME}: $(OBJ)
	@echo ".........linking " ${EXECUTABLE_NAME}
	@echo ..calling $(LD) -o ../${EXECUTABLE_NAME} $(LDFLAGS) $(OBJ) -L./ >${DEBUG_OUTPUT}
	@$(LD) -o ../${EXECUTABLE_NAME} $(LDFLAGS) $(OBJ) -L./

.SUFFIXES : .o .f90 .F90

VPATH=$OBJDIR

%.o: %.F90
	@echo ..........compiling $< in $(CURDIR)
	@echo ..calling $(FC) $(FFLAGS) -c $< -o $@ >${DEBUG_OUTPUT}
	@$(FC) $(FFLAGS) -c $< -o $@

%.o: %.f90
	@echo ..........compiling $< in $(CURDIR)
	@echo ..calling $(FC) $(FFLAGS) -c $< -o $@ >${DEBUG_OUTPUT}
	@$(FC) $(FFLAGS) -c $< -o $@

#######################################################################################
#                                                                                     #
# IT IS ADVISABLE NOT TO EDIT THIS MAKEFILE ABOVE THIS LINE.                          #
# PLEASE NOTE THAT THIS MAKEFILE WILL OPERATE ON A FLAT SINGLE DIRECTORY VERSION      #
# OF THE CODEBASE, THEREFORE vpath / VPATH SHOULD NOT BE NECESSARY HERE               #
#                                                                                     #
#######################################################################################

helper_functions.o: storage_order.F90
helper_functions_gpu.o: storage_order.F90
time_profiling.o: helper_functions.o

# ADD YOUR DEPENDENCIES BELOW THIS LINE
# -------->

main.o: storage_order.F90 helper_functions_gpu.o


