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

#***************************************************************************#
#  Makefile that gets copied into the individual build directories.         #
#  Dependencies of your sources need to be specified here.                  #
#                                                                           #
#  Date             2013/02/01                                              #
#  Author           Michel Müller (RIKEN)                                   #
#***************************************************************************#
TOOLSDIR=${HF_DIR}/hf/
include ./MakesettingsGeneral
include ./Makesettings
include ${TOOLSDIR}MakefileForCompilationCommon

time_profiling.o: helper_functions.o
helper_functions_gpu: helper_functions.o

# ADD YOUR DEPENDENCIES BELOW THIS LINE
# -------->

example.o: ${USER_LIB}

