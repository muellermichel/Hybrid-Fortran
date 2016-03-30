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

import weakref, copy, re
from tools.commons import enum
from tools.metadata import getArguments
from tools.patterns import RegExPatterns
from machinery.commons import ConversionOptions, getSymbolAccessStringAndReminder, purgeDimensionAndGetAdjustedLine
from symbol import DeclarationType

RegionType = enum(
	"MODULE_DECLARATION",
	"KERNEL_CALLER_DECLARATION",
	"OTHER"
)

def implementMatch(line, match, symbol, iterators=[], parallelRegionTemplate=None, callee=None):
	isPointerAssignment = RegExPatterns.Instance().pointerAssignmentPattern.match(line) != None
	argumentString = match.group(3)
	symbolAccessString, remainder = getSymbolAccessStringAndReminder(
		symbol,
		iterators,
		parallelRegionTemplate,
		argumentString,
		callee,
		isPointerAssignment
	)
	patterns = RegExPatterns.Instance()
	pattern1 = r"((?:.|\n)*?(?:\W|^))" + re.escape(symbol.nameInScope()) + re.escape(argumentString) + r"\s*"
	currMatch = patterns.get(pattern1).match(line)
	if not currMatch:
		pattern2 = r"((?:.|\n)*?(?:\W|^))" + re.escape(symbol.name) + re.escape(argumentString) + r"\s*"
		currMatch = patterns.get(pattern2).match(line)
		if not currMatch:
			raise Exception("""\
Symbol %s is accessed in an unexpected way. Note: '_d' postfix is reserved for internal use.
Cannot match one of the following patterns:
pattern1: '%s'
pattern2: '%s'
line:
%s\
""" %(symbol.name, pattern1, pattern2, line))
	prefix = currMatch.group(1)
	return (prefix + symbolAccessString + remainder).strip() + "\n"

def implementLine(line, symbols, parentRoutine, iterators=[], parallelRegionTemplate=None, callee=None):
	adjustedLine = line
	for symbol in symbols:
		symbolWasMatched = False
		lineSections = []
		work = adjustedLine
		nextMatch = symbol.namePattern.match(work)
		while nextMatch:
			symbolWasMatched = True
			prefix = nextMatch.group(1)
			lineSections.append(prefix)
			processed = implementMatch(work, nextMatch, symbol, iterators, parallelRegionTemplate, callee)
			adjustedMatch = symbol.namePattern.match(processed)
			if not adjustedMatch:
				raise Exception("Symbol %s can't be matched again after adjustment. Adjusted portion: %s" %(
					symbol.name,
					processed
				))
			lineSections.append(adjustedMatch.group(2))
			work = adjustedMatch.group(3)
			nextMatch = symbol.namePattern.match(work)
		if not symbolWasMatched:
			raise Exception("symbol %s expected on line '%s' for %s - no match" %(
				symbol.name,
				line,
				parentRoutine.name
			))
		#whatever is left now as "work" is the unmatched trailer of the line
		lineSections.append(work)
		#rebuild adjusted line - next symbol starts adjustment anew
		adjustedLine = "".join(lineSections).strip()
	return adjustedLine

class Region(object):
	def __init__(self, routine):
		self._linesAndSymbols = []
		self._parentRegion = None
		self.loadParentRoutine(routine)

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
		region._linesAndSymbols = copy.copy(self._linesAndSymbols)
		if self._parentRegion:
			region.loadParentRegion(self._parentRegion())
		return region

	def loadParentRegion(self, region):
		self._parentRegion = weakref.ref(region)

	def loadParentRoutine(self, routine):
		self._routineRef = weakref.ref(routine)

	def loadLine(self, line, symbolsOnCurrentLine=None):
		stripped = line.strip()
		if stripped == "":
			return
		self._linesAndSymbols.append((
			stripped,
			symbolsOnCurrentLine
		))

	def implemented(self, skipDebugPrint=False):
		parentRoutine = self._routineRef()
		parallelRegionTemplate = None
		if self._parentRegion:
			parentRegion = self._parentRegion()
			if isinstance(parentRegion, ParallelRegion):
				parallelRegionTemplate = parentRegion.template
		iterators = parentRoutine.implementation.getIterators(parallelRegionTemplate) \
			if parallelRegionTemplate else []
		text = "\n".join([
			implementLine(line, symbols, parentRoutine, iterators, parallelRegionTemplate)
			for (line, symbols) in self._linesAndSymbols
		])
		if text == "":
			return ""
		return self._sanitize(text, skipDebugPrint)

class CallRegion(Region):
	def __init__(self, routine):
		super(CallRegion, self).__init__(routine)
		self._callee = None
		self._passedInSymbolsByName = None
		self._passedInSymbolsByNameInScope = None

	def _adjustedArguments(self, arguments):
		def adjustArgument(argument, parallelRegionTemplate, iterators):
			symbolMatch = RegExPatterns.Instance().symbolNamePattern.match(argument)
			if not symbolMatch:
				return argument
			symbol = self._passedInSymbolsByNameInScope.get(symbolMatch.group(1))
			if not symbol:
				return argument
			symbolAccessString, remainder = getSymbolAccessStringAndReminder(
				symbol,
				iterators,
				parallelRegionTemplate,
				symbolMatch.group(2),
				self._callee
			)
			return symbolAccessString + remainder

		parallelRegionTemplate = None
		if self._parentRegion and isinstance(self._parentRegion(), ParallelRegion):
			parallelRegionTemplate = self._parentRegion().template
		iterators = self._callee.implementation.getIterators(parallelRegionTemplate) \
			if parallelRegionTemplate else []
		return [
			adjustArgument(argument, parallelRegionTemplate, iterators)
			for argument in arguments
		]

	def loadCallee(self, callee):
		self._callee = callee

	def loadPassedInSymbolsByName(self, symbolsByName):
		self._passedInSymbolsByName = copy.copy(symbolsByName)
		self._passedInSymbolsByNameInScope = dict(
			(symbol.nameInScope(), symbol)
			for symbol in symbolsByName.values()
		)

	def clone(self):
		raise NotImplementedError()

	def implemented(self, skipDebugPrint=False):
		if not self._callee:
			raise Exception("call not loaded for call region in %s" %(self._routineRef().name))

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
			numOfProgrammerSpecifiedArguments = len(self._callee.programmerArguments)
			for symbolNum, symbol in enumerate(self._callee.additionalArgumentSymbols):
				hostName = symbol.nameInScope()
				text += hostName
				if symbolNum < len(self._callee.additionalArgumentSymbols) - 1 or numOfProgrammerSpecifiedArguments > 0:
					text += ", %s (type %i) %s" %(bridgeStr1, symbol.declarationType, bridgeStr2)
		text += ", ".join(self._adjustedArguments(self._callee.programmerArguments)) + ")\n"

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
		self._currRegion = Region(routine)
		self._subRegions = [self._currRegion]
		self._currRegion.loadParentRegion(self)
		self._activeTemplate = None
		super(ParallelRegion, self).__init__(routine)

	@property
	def currRegion(self):
		return self._currRegion

	@property
	def template(self):
		return self._activeTemplate

	def switchToRegion(self, region):
		self._currRegion = region
		self._subRegions.append(region)
		region.loadParentRegion(self)

	def loadLine(self, line, symbolsOnCurrentLine=None):
		self._currRegion.loadLine(line, symbolsOnCurrentLine)

	def loadActiveParallelRegionTemplate(self, templateNode):
		self._activeTemplate = templateNode

	def loadParentRoutine(self, routine):
		super(ParallelRegion, self).loadParentRoutine(routine)
		for region in self._subRegions:
			region.loadParentRoutine(routine)

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
	def __init__(self, routine):
		super(RoutineSpecificationRegion, self).__init__(routine)
		self._additionalParametersByKernelName = None
		self._packedRealSymbolsByCalleeName = None
		self._symbolsToAdd = None
		self._compactionDeclarationPrefixByCalleeName = None
		self._currAdditionalCompactedSubroutineParameters = None
		self._allImports = None

	def clone(self):
		clone = super(RoutineSpecificationRegion, self).clone()
		clone.loadAdditionalContext(
			self._additionalParametersByKernelName,
			self._packedRealSymbolsByCalleeName,
			self._symbolsToAdd,
			self._compactionDeclarationPrefixByCalleeName,
			self._currAdditionalCompactedSubroutineParameters,
			self._allImports
		)
		return clone

	def loadAdditionalContext(
		self,
		additionalParametersByKernelName,
		packedRealSymbolsByCalleeName,
		symbolsToAdd,
		compactionDeclarationPrefixByCalleeName,
		currAdditionalCompactedSubroutineParameters,
		allImports
	):
		self._additionalParametersByKernelName = copy.copy(additionalParametersByKernelName)
		self._packedRealSymbolsByCalleeName = copy.copy(packedRealSymbolsByCalleeName)
		self._symbolsToAdd = copy.copy(symbolsToAdd)
		self._compactionDeclarationPrefixByCalleeName = copy.copy(compactionDeclarationPrefixByCalleeName)
		self._currAdditionalCompactedSubroutineParameters = copy.copy(currAdditionalCompactedSubroutineParameters)
		self._allImports = copy.copy(allImports)

	def implemented(self, skipDebugPrint=False):
		def getImportLine(line, importedSymbols, parentRoutine):
			return parentRoutine.implementation.adjustImportForDevice(
				line,
				importedSymbols,
				RegionType.KERNEL_CALLER_DECLARATION if parentRoutine.isCallingKernel else RegionType.OTHER,
				parentRoutine.node.getAttribute('parallelRegionPosition'),
				parentRoutine.parallelRegionTemplates
			).strip() + "\n"

		parentRoutine = self._routineRef()
		declarationRegionType = RegionType.OTHER
		if parentRoutine.isCallingKernel:
			declarationRegionType = RegionType.KERNEL_CALLER_DECLARATION

		if self._additionalParametersByKernelName == None \
		or self._packedRealSymbolsByCalleeName == None \
		or self._symbolsToAdd == None \
		or self._compactionDeclarationPrefixByCalleeName == None \
		or self._currAdditionalCompactedSubroutineParameters == None:
			raise Exception("additional context not properly loaded for routine specification region in %s" %(
				parentRoutine.name
			))

		importedSymbols = []
		declaredSymbols = []
		textBeforeImports = ""
		textBeforeDeclarations = ""
		textAfterDeclarations = ""
		for (line, symbols) in self._linesAndSymbols:
			if not symbols or len(symbols) == 0:
				if len(importedSymbols) == 0:
					textBeforeImports += line.strip() + "\n"
				elif len(declaredSymbols) == 0:
					textBeforeDeclarations += line.strip() + "\n"
				else:
					textAfterDeclarations += line.strip() + "\n"
				continue
			for symbol in symbols:
				match = symbol.getDeclarationMatch(line)
				if match:
					declaredSymbols.append(symbol)
					continue
				match = symbol.importPattern.match(line)
				if not match:
					match = symbol.importMapPattern.match(line)
				if match:
					importedSymbols.append(symbol)
					continue
				raise Exception("symbol %s expected to be referenced in line '%s', but all matchings have failed" %(
					symbol.name,
					line
				))

		text = textBeforeImports
		importsRequiredDict = copy.copy(self._allImports)
		if len(importedSymbols) > 0:
			text += getImportLine(None, importedSymbols, parentRoutine)
			if importsRequiredDict:
				for symbol in importedSymbols:
					k = (symbol.sourceModule, symbol.nameInScope())
					if k in importsRequiredDict:
						del importsRequiredDict[k]
		if importsRequiredDict:
			for (sourceModule, nameInScope) in importsRequiredDict:
				sourceName = importsRequiredDict[(sourceModule, nameInScope)]
				text += getImportLine(
					"use %s, only: %s => %s" %(sourceModule, nameInScope, sourceName),
					[],
					parentRoutine
				)

		text += textBeforeDeclarations
		if len(declaredSymbols) > 0:
			text += "\n".join([
				parentRoutine.implementation.adjustDeclarationForDevice(
					symbol.getDeclarationLine(purgeList=[]),
					[symbol],
					declarationRegionType,
					parentRoutine.node.getAttribute('parallelRegionPosition')
				).strip()
				for symbol in declaredSymbols
			]).strip() + "\n"
		text += textAfterDeclarations

		numberOfAdditionalDeclarations = (
			len(sum([
				self._additionalParametersByKernelName[kname][1]
				for kname in self._additionalParametersByKernelName
			], [])) + len(self._symbolsToAdd) + len(self._packedRealSymbolsByCalleeName.keys())
		)
		if numberOfAdditionalDeclarations > 0:
			text += "\n! ****** additional symbols inserted by framework to emulate device support of language features\n"
		defaultPurgeList = ['intent', 'public', 'parameter', 'allocatable']
		for symbol in self._symbolsToAdd:
			purgeList = defaultPurgeList
			if not symbol.isCompacted:
				purgeList=['public', 'parameter', 'allocatable']
			text += parentRoutine.implementation.adjustDeclarationForDevice(
				symbol.getDeclarationLine(purgeList).strip(),
				[symbol],
				declarationRegionType,
				parentRoutine.node.getAttribute('parallelRegionPosition')
			).rstrip() + " ! type %i symbol added for this subroutine\n" %(symbol.declarationType)
		for callee in parentRoutine.callees:
			additionalImports, additionalDeclarations = self._additionalParametersByKernelName.get(
				callee.name,
				([], [])
			)
			additionalImportSymbolsByName = {}
			for symbol in additionalImports:
				additionalImportSymbolsByName[symbol.name] = symbol

			implementation = callee.implementation
			for symbol in parentRoutine.filterOutSymbolsAlreadyAliveInCurrentScope(additionalDeclarations):
				if symbol.declarationType not in [DeclarationType.LOCAL_ARRAY, DeclarationType.LOCAL_SCALAR]:
					# only symbols that are local to the kernel actually need to be declared here.
					# Everything else we should have in our own scope already, either through additional imports or
					# through module association (we assume the kernel and its wrapper reside in the same module)
					continue

				#in case the array uses domain sizes in the declaration that are additional symbols themselves
				#we need to fix them.
				adjustedDomains = []
				for (domName, domSize) in symbol.domains:
					domSizeSymbol = additionalImportSymbolsByName.get(domSize)
					if domSizeSymbol is None:
						adjustedDomains.append((domName, domSize))
						continue
					adjustedDomains.append((domName, domSizeSymbol.nameInScope()))
				symbol.domains = adjustedDomains

				text += implementation.adjustDeclarationForDevice(
					symbol.getDeclarationLine(defaultPurgeList).strip(),
					[symbol],
					declarationRegionType,
					parentRoutine.node.getAttribute('parallelRegionPosition')
				).rstrip() + " ! type %i symbol added for callee %s\n" %(symbol.declarationType, callee.name)
			toBeCompacted = self._packedRealSymbolsByCalleeName.get(callee.name, [])
			if len(toBeCompacted) > 0:
				#TODO: generalize for cases where we don't want this to be on the device
				#(e.g. put this into Implementation class)
				compactedArrayName = "hfimp_%s" %(callee.name)
				compactedArray = FrameworkArray(
					compactedArrayName,
					self._compactionDeclarationPrefixByCalleeName[callee.name],
					domains=[("hfauto", str(len(toBeCompacted)))],
					isOnDevice=True
				)
				text += implementation.adjustDeclarationForDevice(
					compactedArray.getDeclarationLine().strip(),
					[compactedArray],
					declarationRegionType,
					parentRoutine.node.getAttribute('parallelRegionPosition')
				).rstrip() + " ! compaction array added for callee %s\n" %(callee.name)

		calleesWithPackedReals = self._packedRealSymbolsByCalleeName.keys()
		for calleeName in calleesWithPackedReals:
			for idx, symbol in enumerate(sorted(self._packedRealSymbolsByCalleeName[calleeName])):
				#$$$ clean this up, the hf_imp prefix should be decided within the symbol class
				text += "hfimp_%s(%i) = %s" %(calleeName, idx+1, symbol.nameInScope()) + \
					 " ! type %i symbol compaction for callee %s\n" %(symbol.declarationType, calleeName)

		for idx, symbol in enumerate(self._currAdditionalCompactedSubroutineParameters):
			#$$$ clean this up, the hf_imp prefix should be decided within the symbol class
			text += "%s = hfimp_%s(%i)" %(symbol.nameInScope(), parentRoutine.name, idx+1) + \
					 " ! additional type %i symbol compaction\n" %(symbol.declarationType)

		if numberOfAdditionalDeclarations > 0:
			text += "! ****** end additional symbols\n\n"

		text += parentRoutine.implementation.declarationEnd(
			parentRoutine.symbolsByName.values() + parentRoutine.additionalImports,
			parentRoutine.isCallingKernel,
			parentRoutine.node,
			parentRoutine.parallelRegionTemplates
		)
		return self._sanitize(text, skipDebugPrint)

class RoutineEarlyExitRegion(Region):
	def implemented(self, skipDebugPrint=False):
		parentRoutine = self._routineRef()
		text = parentRoutine.implementation.subroutineExitPoint(
			parentRoutine.symbolsByName.values(),
			parentRoutine.isCallingKernel,
			isSubroutineEnd=False
		)
		text += super(RoutineExitRegion, self).implemented(skipDebugPrint=True)

		return self._sanitize(text, skipDebugPrint)