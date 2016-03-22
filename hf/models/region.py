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

import weakref, copy
from tools.commons import enum
from tools.metadata import getArguments
from machinery.commons import ConversionOptions

RegionType = enum(
	"MODULE_DECLARATION",
	"KERNEL_CALLER_DECLARATION",
	"OTHER"
)

class Region(object):
	def __init__(self, routine):
		self._linesAndSymbols = []
		self._routineRef = weakref.ref(routine)

	@property
	def parentRoutine(self):
		return self._routineRef()

	@parentRoutine.setter
	def parentRoutine(self, _routine):
		self._routineRef = weakref.ref(_routine)

	def _sanitize(self, text, skipDebugPrint=False):
		if not ConversionOptions.Instance().debugPrint or skipDebugPrint:
			return text.strip() + "\n"
		return "!<--- %s\n%s\n!--->\n" %(
			type(self),
			text.strip()
		)

	def clone(self):
		region = self.__class__(self.parentRoutine)
		region._linesAndSymbols = copy.deepcopy(self._linesAndSymbols)
		return region

	def loadLine(self, line, symbolsOnCurrentLine=None):
		stripped = line.strip()
		if stripped == "":
			return
		self._linesAndSymbols.append((
			stripped,
			symbolsOnCurrentLine
		))

	def implemented(self, skipDebugPrint=False):
		text = "\n".join([line for (line, symbols) in self._linesAndSymbols])
		if text == "":
			return ""
		return self._sanitize(text, skipDebugPrint)

class CallRegion(Region):
	def __init__(self, routine):
		super(CallRegion, self).__init__(routine)
		self._callee = None
		self._passedInSymbolsByName = None

	def loadCallee(self, callee):
		self._callee = callee

	def loadPassedInSymbolsByName(self, symbolsByName):
		self._passedInSymbolsByName = copy.copy(symbolsByName)

	def clone(self):
		raise NotImplementedError()

	def implemented(self, skipDebugPrint=False):
		if not self._callee:
			raise Exception("call needs to be loaded at this point")

		text = ""
		argumentSymbols = None
		#this hasattr is used to test the callee for analyzability without circular imports
		if hasattr(self._callee, "implementation"):
			#$$$ we could have an ordering problem with _passedInSymbolsByName
			argumentSymbols = self._callee.additionalArgumentSymbols + self._passedInSymbolsByName.values()
			for symbol in argumentSymbols:
				text += self._callee.implementation.callPreparationForPassedSymbol(
					self._routineRef().node,
					symbolInCaller=symbol
				)

		parallelRegionPosition = None
		if hasattr(self._callee, "implementation"):
			parallelRegionPosition = self._callee.node.getAttribute("parallelRegionPosition")
		if hasattr(self._callee, "implementation") and parallelRegionPosition == "within":
			if not self._callee.parallelRegionTemplates \
			or len(self._callee.parallelRegionTemplates) == 0:
				raise Exception("No parallel region templates found for subroutine %s" %(
					self._callee.name
				))
			text += "%s call %s %s" %(
				self._callee.implementation.kernelCallPreparation(
					self._callee.parallelRegionTemplates[0],
					calleeNode=self._callee.node
				),
				self._callee.name,
				self._callee.implementation.kernelCallConfig()
			)
		else:
			text += "call " + self._callee.name

		text += "("
		if hasattr(self._callee, "implementation"):
			if len(self._callee.additionalArgumentSymbols) > 0:
				text += " &\n"
			bridgeStr1 = " & !additional parameter"
			bridgeStr2 = "inserted by framework\n& "
			numOfProgrammerSpecifiedArguments = len(self._callee.programmerArgumentNames)
			for symbolNum, symbol in enumerate(self._callee.additionalArgumentSymbols):
				hostName = symbol.nameInScope()
				text += hostName
				if symbolNum < len(self._callee.additionalArgumentSymbols) - 1 or numOfProgrammerSpecifiedArguments > 0:
					text += ", %s (type %i) %s" %(bridgeStr1, symbol.declarationType, bridgeStr2)
		text += ", ".join(self._callee.programmerArgumentNames) + ")\n"

		if hasattr(self._callee, "implementation"):
			allSymbolsPassedByName = dict(
				(symbol.name, symbol)
				for symbol in argumentSymbols
			)
			text += self._callee.implementation.kernelCallPost(
				allSymbolsPassedByName,
				self._callee.node
			)
			for symbol in argumentSymbols:
				text += self._callee.implementation.callPostForPassedSymbol(
					self._routineRef().node,
					symbolInCaller=symbol
				)
		return self._sanitize(text, skipDebugPrint)

class ParallelRegion(Region):
	def __init__(self, routine):
		super(ParallelRegion, self).__init__(routine)
		self._currRegion = Region(routine)
		self._subRegions = [self._currRegion]
		self._activeTemplate = None

	@property
	def template(self):
		return self._activeTemplate

	def switchToRegion(self, region):
		self._currRegion = region
		self._subRegions.append(region)

	def loadLine(self, line, symbolsOnCurrentLine=None):
		self._currRegion.loadLine(line, symbolsOnCurrentLine)

	def loadActiveParallelRegionTemplate(self, templateNode):
		self._activeTemplate = templateNode

	def clone(self):
		raise NotImplementedError()

	def implemented(self, skipDebugPrint=False):
		parentRoutine = self._routineRef()
		hasParallelRegionWithin = parentRoutine.node.getAttribute('parallelRegionPosition') == 'within'
		if hasParallelRegionWithin \
		and not self._activeTemplate:
			raise Exception("cannot implement parallel region without a template node loaded")

		text = ""
		if hasParallelRegionWithin:
			text += parentRoutine.implementation.parallelRegionBegin(
				parentRoutine.symbolsByName.values(),
				self._activeTemplate
			).strip() + "\n"
		text += "\n".join([region.implemented() for region in self._subRegions])
		if hasParallelRegionWithin:
			text += parentRoutine.implementation.parallelRegionEnd(
				self._activeTemplate
			).strip() + "\n"
		return self._sanitize(text, skipDebugPrint)

class RoutineSpecificationRegion(Region):
	def implemented(self, skipDebugPrint=False):
		text = super(RoutineSpecificationRegion, self).implemented(skipDebugPrint=True)
		parentRoutine = self._routineRef()
		text += parentRoutine.implementation.declarationEnd(
			parentRoutine.symbolsByName.values() + parentRoutine.additionalImports,
			parentRoutine.isCallingKernel,
			parentRoutine.node,
			parentRoutine.parallelRegionTemplates
		)
		return self._sanitize(text, skipDebugPrint)