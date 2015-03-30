# Copyright (C) 2015 Michel Müller, Tokyo Institute of Technology

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
#  Makefile to create an example project and run tests.                     #
#                                                                           #
#  Date             2014/07/11                                              #
#  Author           Michel Müller (Titech)                                  #
#                                                                           #
#***************************************************************************#
SHELL=/bin/bash
TOOLSDIR=${HF_DIR}/hf_processor/
EXAMPLEDIR=${HF_DIR}/example/
EXAMPLEDIR_SOURCE=${EXAMPLEDIR}source/
EXAMPLEDIR_CONFIG=${EXAMPLEDIR}config/

TEST_PROJECTS=examples/simple_stencil examples/stencil_with_local_array examples/stencil_with_passed_in_scalar_from_array examples/array_accessor_functions examples/early_returns examples/mixed_implementations examples/tracing examples/strides examples/simple_openACC examples/branches_with_openACC examples/module_data_with_openACC examples/openACC_hybrid_hostonly examples/poisson2d_fem_iterative examples/midaco_solver examples/diffusion3d examples/particle
ADDITIONAL_TEST_PROJECTS=pp

TEST_TARGETS=$(addprefix test_,$(TEST_PROJECTS))
CLEAN_TARGETS=$(addprefix clean_,$(TEST_PROJECTS))
ADDITIONAL_TEST_TARGETS=$(addprefix test_,$(ADDITIONAL_TEST_PROJECTS))
ALL_TEST_PROJECTS=${TEST_PROJECTS} ${ADDITIONAL_TEST_PROJECTS}

.PHONY: all example tests ${TEST_TARGETS} ${ADDITIONAL_TEST_TARGETS}

all: example

tests: test_example ${TEST_TARGETS}

clean: clean_example ${CLEAN_TARGETS}

#note: cp -n does not exist in older GNU utils, so we emulate it here for compatibility
example:
	@mkdir -p ${HF_DIR}/example
	@mkdir -p ${EXAMPLEDIR_CONFIG}
	@mkdir -p ${EXAMPLEDIR_SOURCE}
	@if [ ! -e ${EXAMPLEDIR}Makefile ]; then \
	    cp -f ${TOOLSDIR}MakefileForProjectTemplate ${EXAMPLEDIR}Makefile; \
	fi; \
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

test_example: example
	@echo "###########################################################################################"
	@echo "########################## attempting to test example #####################################"
	@echo "###########################################################################################"
	@cd example && make clean
	@cd example && make tests

define test_rules
  test_$(1):
	@echo "###########################################################################################"
	@echo "########################## attempting to test $(1) ###############################"
	@echo "###########################################################################################"
	@cd $(1) && make clean
	@cd $(1) && make tests
endef

clean_example:
	@cd example && make clean

define clean_rules
  clean_$(1):
	@cd $(1) && make clean
endef

$(foreach project,$(ALL_TEST_PROJECTS),$(eval $(call test_rules,$(project))))
$(foreach project,$(ALL_TEST_PROJECTS),$(eval $(call clean_rules,$(project))))
