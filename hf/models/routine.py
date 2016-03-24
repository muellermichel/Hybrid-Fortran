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
from models.region import RegionType, RoutineSpecificationRegion, ParallelRegion, CallRegion
from machinery.commons import ConversionOptions

def uniqueIdentifier(routineName, implementationName):
	return (routineName + "_hfauto_" + implementationName).strip()

class Routine(object):
	def __init__(self, name):
		if not type(name) in [str, unicode] or name.strip() == "":
			raise Exception("no valid name passed when trying to initialize routine")
		self.name = name
		self._programmerArguments = None

	@property
	def programmerArguments(self):
		if self._programmerArguments == None:
			raise Exception("programmer arguments not yet loaded for %s" %(self.name))
		return self._programmerArguments

	def loadArguments(self, arguments):
		self._programmerArguments = copy.copy(arguments)

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
		self.parallelRegionTemplates = copy.copy(parallelRegionTemplates)
		self.symbolsByName = None
		self.callsByCalleeName = {}
		self._currRegion = RoutineSpecificationRegion(self)
		self._regions = [self._currRegion]
		self._additionalArguments = None
		self._additionalImports = None
		self._symbolsToUpdate = None

	@property
	def additionalArgumentSymbols(self):
		if self._additionalArguments == None:
			raise Exception("additional arguments not yet loaded for %s" %(self.name))
		return self._additionalArguments

	@property
	def additionalImports(self):
		return self._additionalImports

	@property
	def currRegion(self):
		return self._currRegion

	@property
	def regions(self):
		return self._regions

	@property
	def isCallingKernel(self):
		for region in self._regions:
			if isinstance(region, CallRegion) \
			and region._callee \
			and region._callee.node.getAttribute("parallelRegionPosition") == "within":
				return True
		return False

	@regions.setter
	def regions(self, _regions):
		self._regions = _regions

	def _checkParallelRegions(self):
		if self.node.getAttribute('parallelRegionPosition') != 'within':
			return
		templates = self.parallelRegionTemplates
		if not templates or len(templates) == 0:
			raise Exception("Unexpected: no parallel template definition found for routine '%s'" %(
				self.name
			))
		if len(templates) > 1 and self.implementation.multipleParallelRegionsPerSubroutineAllowed != True:
			raise Exception("Unexpected: more than one parallel region templates found for subroutine '%s' \
containing a parallelRegion directive. \
This is not allowed for implementations using %s.\
				" %(
					self.name,
					type(self.implementation).__name__
				)
			)

	def _updateSymbolState(self):
		if self._symbolsToUpdate == None:
			raise Exception("no symbols loaded for updating in routine %s" %(self.name))
		regionType = RegionType.KERNEL_CALLER_DECLARATION if self.isCallingKernel else RegionType.OTHER
		for symbol in self._symbolsToUpdate:
			self.implementation.updateSymbolDeviceState(
				symbol,
				regionType,
				self.node.getAttribute("parallelRegionPosition")
			)

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

	def filterOutSymbolsAlreadyAliveInCurrentScope(self, symbolList):
		return [
			symbol for symbol in symbolList
			if not symbol.analysis \
			or ( \
				symbol.uniqueIdentifier not in self.symbolsByName \
				and symbol.analysis.argumentIndexByRoutineName.get(self.name, -1) == -1 \
			)
		]

	def nameInScope(self):
		if not self.sisterRoutine:
			return self.name
		return uniqueIdentifier(self.name, self.implementation.architecture[0])

	def createCloneWithMetadata(self, name):
		clone = AnalyzableRoutine(
			name,
			routineNode=self.node.cloneNode(deep=True),
			parallelRegionTemplates=copy.copy(self.parallelRegionTemplates),
			implementation=self.implementation
		)
		clone._programmerArguments = copy.copy(self._programmerArguments)
		clone._additionalArguments = copy.copy(self._additionalArguments)
		clone._additionalImports = copy.copy(self._additionalImports)
		clone._symbolsToUpdate = copy.copy(self._symbolsToUpdate)
		clone.symbolsByName = copy.copy(self.symbolsByName)
		clone.callsByCalleeName = copy.copy(self.callsByCalleeName)
		return clone

	def resetRegions(self, firstRegion):
		self._regions = []
		self.addRegion(firstRegion)

	def createRegion(self, regionClassName="Region", oldRegion=None):
		import models.region
		regionClass = getattr(models.region, regionClassName)
		region = regionClass(self)
		if isinstance(self._currRegion, ParallelRegion) \
		and not isinstance(oldRegion, ParallelRegion):
			self._currRegion.switchToRegion(region)
		else:
			self.addRegion(region)
		return region

	def addRegion(self, region):
		self._regions.append(region)
		self._currRegion = region
		region.loadParentRoutine(self)

	def loadAdditionalArgumentSymbols(self, argumentSymbols):
		self._additionalArguments = copy.copy(argumentSymbols)

	def loadAdditionalImportSymbols(self, importSymbols):
		self._additionalImports = copy.copy(importSymbols)

	def loadSymbolsByName(self, symbolsByName):
		self.symbolsByName = copy.copy(symbolsByName)

	def loadSymbolsToUpdate(self, symbolsToUpdate):
		self._symbolsToUpdate = copy.copy(symbolsToUpdate)

	def loadCall(self, callRoutine):
		self.callsByCalleeName[callRoutine.name] = callRoutine
		if isinstance(self._currRegion, CallRegion):
			self._currRegion.loadCallee(callRoutine)

	def loadLine(self, line, symbolsOnCurrentLine=None):
		self._currRegion.loadLine(line, symbolsOnCurrentLine)

	def implemented(self):
		self._checkParallelRegions()
		self._updateSymbolState()
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