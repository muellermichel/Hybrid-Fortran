#!/usr/bin/python
# -*- coding: UTF-8 -*-

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

import copy
from models.region import RegionType, RoutineSpecificationRegion, ParallelRegion
from machinery.commons import ConversionOptions

def uniqueIdentifier(routineName, implementationName):
	return (routineName + "_hfauto_" + implementationName).strip()

class Routine(object):
	def __init__(self, name):
		if not type(name) in [str, unicode] or name.strip() == "":
			raise Exception("no valid name passed when trying to initialize routine")
		self.name = name

	def nameInScope(self):
		return self.name

class AnalyzableRoutine(Routine):
	def __init__(self, name, routineNode, parallelRegionTemplates, implementation):
		super(AnalyzableRoutine, self).__init__(name)
		if not routineNode:
			raise Exception("no definition passed when trying to initialize routine '%s'" %(name))
		if not implementation:
			raise Exception("no implementation passed when trying to initialize routine '%s'" %(name))
		self.name = name
		self.implementation = implementation
		self.sisterRoutine = None
		self.node = routineNode
		self.parallelRegionTemplates = parallelRegionTemplates
		self.symbolsByName = None
		self.callsByCalleeName = {}
		self.isCallingKernel = False
		self._currRegion = RoutineSpecificationRegion(self)
		self._regions = [self._currRegion]
		self._programmerArguments = None
		self._additionalArguments = None
		self._additionalImports = None

	@property
	def programmerArgumentNames(self):
		if self._programmerArguments == None:
			raise Exception("programmer arguments not yet loaded for %s" %(self.name))
		return self._programmerArguments

	@property
	def additionalArgumentSymbols(self):
		if self._additionalArguments == None:
			raise Exception("additional arguments not yet loaded for %s" %(self.name))
		return self._additionalArguments

	def _implementHeader(self):
		parameterList = ""
		if self._additionalArguments and len(self._additionalArguments) > 0:
			parameterList += "&\n&"
			parameterList += "&, ".join([
				"%s & !additional type %i symbol inserted by framework \n" %(
					symbol.nameInScope(),
					symbol.declarationType
				)
				for symbol in self._additionalArguments
			])
		if self._additionalArguments and len(self._additionalArguments) > 0 \
		and self._programmerArguments and len(self._programmerArguments) > 0:
			parameterList += "&, "
		elif self._additionalArguments and len(self._additionalArguments) > 0:
			parameterList += "& "
		if self._programmerArguments:
			parameterList += ", ".join(self._programmerArguments)
		return "%s subroutine %s(%s)\n" %(
			self.implementation.subroutinePrefix(self.node),
			self.name,
			parameterList
		)

	def _implementAdditionalImports(self):
		if not self._additionalImports or len(self._additionalImports) == 0:
			return self.implementation.additionalIncludes()
		return self.implementation.adjustImportForDevice(
            None,
            self._additionalImports,
            RegionType.KERNEL_CALLER_DECLARATION if self.isCallingKernel else RegionType.OTHER,
            self.node.getAttribute('parallelRegionPosition'),
            self.parallelRegionTemplates
        ) + self.implementation.additionalIncludes()

	def _implementFooter(self):
		return self.implementation.subroutineExitPoint(
            self.symbolsByName.values(),
            self.isCallingKernel,
            isSubroutineEnd=True
        ) + "end subroutine\n"

	def nameInScope(self):
		if not self.sisterRoutine:
			return self.name
		return uniqueIdentifier(self.name, self.implementation.architecture[0])

	def createRegion(self, regionClassName="Region", oldRegion=None):
		import models.region
		regionClass = getattr(models.region, regionClassName)
		region = regionClass(self)
		if isinstance(self._currRegion, ParallelRegion) \
		and not isinstance(oldRegion, ParallelRegion):
			self._currRegion.switchToRegion(region)
		else:
			self._regions.append(region)
			self._currRegion = region
		return region

	def loadArguments(self, arguments):
		self._programmerArguments = copy.copy(arguments)

	def loadAdditionalArgumentSymbols(self, argumentSymbols):
		self._additionalArguments = copy.copy(argumentSymbols)

	def loadAdditionalImportSymbols(self, importSymbols):
		self._additionalImports = copy.copy(importSymbols)

	def loadSymbolsByName(self, symbolsByName):
		self.symbolsByName = copy.copy(symbolsByName)

	def loadCall(self, callRoutine):
		self.callsByCalleeName[callRoutine.name] = callRoutine
		if callRoutine.node.getAttribute("parallelRegionPosition") == "within":
			self.isCallingKernel = True

	def loadLine(self, line):
		self._currRegion.loadLine(line)

	def implemented(self):
		implementedRoutineElements = [self._implementHeader(), self._implementAdditionalImports()]
		implementedRoutineElements += [region.implemented() for region in self._regions]
		implementedRoutineElements += [self._implementFooter()]
		purgedRoutineElements = [
			(index, text) for index, text in enumerate(implementedRoutineElements)
			if text != ""
		]
		return "\n".join([
			text
			for (index, text) in purgedRoutineElements
		])