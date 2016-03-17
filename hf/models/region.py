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

import weakref
from tools.commons import enum
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

	def loadLine(self, line, symbolsOnCurrentLine=None):
		stripped = line.strip()
		if stripped == "":
			return
		self._linesAndSymbols.append((
			stripped,
			symbolsOnCurrentLine
		))

	def implemented(self):
		text = "\n".join([line for (line, symbols) in self._linesAndSymbols])
		if text == "":
			return ""
		result = ""
		if ConversionOptions.Instance().debugPrint:
			result += "!<--- %s\n" %(type(self))
		result += text + "\n"
		if ConversionOptions.Instance().debugPrint:
			result += "!--->\n"
		return result

class CallRegion(Region):
	def __init__(self, routine):
		super(CallRegion, self).__init__(routine)
		self._callee = None

	def loadCallee(self, callee):
		self._callee = callee

	def implemented(self):
		if not self._callee:
			raise Exception("call needs to be loaded at this point")

		text = ""
		argumentSymbols = self._callee._additionalArguments + self._callee._programmerArguments
		for symbol in argumentSymbols:
			text += self._callee.implementation.callPreparationForPassedSymbol(
				self._routineRef().node,
				symbolInCaller=symbol
			)

		parallelRegionPosition = None
		if isinstance(self._callee, AnalyzableRoutine):
			parallelRegionPosition = self._callee.node.getAttribute("parallelRegionPosition")
		if isinstance(self._callee, AnalyzableRoutine) and parallelRegionPosition == "within":
			if not self._callee.parallelRegionTemplates \
			or len(self._callee.parallelRegionTemplates) == 0:
				raise Exception("No parallel region templates found for subroutine %s" %(
					self._callee.name
				))
			text += "%s call %s %s" %(
				self._callee.implementation.kernelCallPreparation(
					parallelRegionTemplates[0],
					calleeNode=self.currCallee.node
				),
				self._callee.name,
				self._callee.implementation.kernelCallConfig()
			)
		else:
			text += "call " + self._callee.name

		if len(self._callee._additionalArguments) > 0:
			text += "( &\n"
		else:
			text += "("
		bridgeStr1 = " & !additional parameter"
		bridgeStr2 = "inserted by framework\n& "
		numOfProgrammerSpecifiedArguments = len(self._callee._programmerArguments)
		for symbolNum, symbol in enumerate(self._callee._additionalArguments):
			hostName = symbol.nameInScope()
			adjustedLine += hostName
			if symbolNum < len(self._callee._additionalArguments) - 1 or numOfProgrammerSpecifiedArguments > 0:
				adjustedLine += ", "
			if symbolNum < len(self._callee._additionalArguments) - 1 or paramListMatch:
				adjustedLine += "%s (type %i) %s" %(bridgeStr1, symbol.declarationType, bridgeStr2)

		text += super(CallRegion, self).implemented()

		if isinstance(self._callee, AnalyzableRoutine):
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
		return text

class ParallelRegion(Region):
	def __init__(self, routine):
		super(CallRegion, self).__init__(routine)
		self._currRegion = Region()
		self._subRegions = [self._currRegion]

	def switchToRegion(self, region):
		self._currRegion = region
		self._subRegions.append(region)

	def loadLine(self, line, symbolsOnCurrentLine=None):
		self._currRegion.loadLine(self, line, symbolsOnCurrentLine)

	def implemented(self):
		return "\n".join([region.implemented() for region in self._subRegions])

class RoutineSpecificationRegion(Region):
	pass