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

import copy, re
from tools.commons import enum, UsageError, OrderedDict
from tools.metadata import getArguments, getIterators
from tools.patterns import regexPatterns
from machinery.commons import conversionOptions, \
	getSymbolAccessStringAndRemainder, \
	implement, \
	replaceEarlyExits
from symbol import DeclarationType, FrameworkArray, frameworkArrayName, limitLength, uniqueIdentifier

RegionType = enum(
	"MODULE_DECLARATION",
	"KERNEL_CALLER_DECLARATION",
	"OTHER"
)

def implementSymbolAccessStringAndRemainder(
	line,
	suffix,
	symbol,
	iterators=[],
	parallelRegionTemplate=None,
	callee=None,
	useDeviceVersionIfAvailable=True
):
	isPointerAssignment = regexPatterns.pointerAssignmentPattern.match(line) != None
	try:
		symbolAccessString, remainder = getSymbolAccessStringAndRemainder(
			symbol,
			iterators,
			parallelRegionTemplate,
			suffix,
			callee,
			isPointerAssignment,
			useDeviceVersionIfAvailable=useDeviceVersionIfAvailable
		)
	except UsageError as e:
		raise UsageError("%s; Print of Line: %s" %(str(e), line))
	return symbolAccessString, remainder

#retuns a new region with some code. does not support symbols being used within the code
def regionWithInertCode(routine, codeLines):
	region = Region(routine)
	for line in codeLines:
		region.loadLine(line)
	return region

class Region(object):
	def __init__(self):
		self._linesAndSymbols = []

	def __contains__(self, text):
		for line, _ in self._linesAndSymbols:
			if text in line:
				return True
		return False

	@property
	def usedSymbolNames(self):
		return set([symbol.name for symbol in sum([symbols for (_, symbols) in self._linesAndSymbols], [])])

	@property
	def linesAndSymbols(self):
		return self._linesAndSymbols

	@property
	def isCallingKernel(self):
		return False

	def _sanitize(self, text, skipDebugPrint=False):
		if not conversionOptions.debugPrint or skipDebugPrint:
			return text.strip() + "\n"
		return "!<--- %s\n%s\n!--->\n" %(
			type(self),
			text.strip()
		)

	def clone(self):
		region = self.__class__()
		region._linesAndSymbols = copy.copy(self._linesAndSymbols)
		return region

	def loadLine(self, line, symbolsOnCurrentLine=None):
		stripped = line.strip()
		if stripped == "":
			return
		self._linesAndSymbols.append((
			stripped,
			symbolsOnCurrentLine if symbolsOnCurrentLine else []
		))

	def firstAccessTypeOfScalar(self, symbol):
		if symbol.domains:
			raise Exception("non scalars not supported for this operation")
		for line, symbols in self._linesAndSymbols:
			if not symbol in symbols:
				continue
			if symbol.scalarWriteAccessPattern.match(line):
				return "w"
			return "r"
		return None

	def implemented(self, parentRoutine, parentRegion=None, skipDebugPrint=False):
		parallelRegionTemplate = None
		if isinstance(parentRegion, ParallelRegion):
			parallelRegionTemplate = parentRegion.template
		iterators = parentRoutine.implementation.getIterators(parallelRegionTemplate) \
			if parallelRegionTemplate else []
		text = "\n".join([
			implement(
				replaceEarlyExits(line, parentRoutine.implementation, parentRoutine.node.getAttribute('parallelRegionPosition')),
				symbols,
				implementSymbolAccessStringAndRemainder,
				iterators,
				parallelRegionTemplate,
				useDeviceVersionIfAvailable=parentRoutine.implementation.onDevice
			)
			for (line, symbols) in self._linesAndSymbols
		])
		if text == "":
			return ""
		return self._sanitize(text, skipDebugPrint)

class CallRegion(Region):
	def __init__(self):
		super(CallRegion, self).__init__()
		self._callee = None
		self._passedInSymbolsByName = None

	@property
	def isCallingKernel(self):
		if self._callee \
		and hasattr(self._callee, "node") \
		and self._callee.node.getAttribute("parallelRegionPosition") == "within":
			return True
		return False

	@property
	def usedSymbolNames(self):
		compactedSymbols = sum([
			 s.compactedSymbols for s in self._callee._additionalArguments
			 if isinstance(s, FrameworkArray)
		], []) if hasattr(self._callee, "_additionalArguments") and self._callee._additionalArguments else []

		additionalArgumentSymbols = [
			s for s in self._callee._additionalArguments
			if not isinstance(s, FrameworkArray)
		] if hasattr(self._callee, "_additionalArguments") and self._callee._additionalArguments else []

		return super(CallRegion, self).usedSymbolNames \
			| set([a.split("(")[0].strip() for a in self._callee.programmerArguments]) \
			| set([s.name for s in compactedSymbols + additionalArgumentSymbols])

	def _adjustedArguments(self, arguments, parentRoutine, parentRegion=None):
		def adjustArgument(argument, parallelRegionTemplate, iterators):
			return implement(
				argument,
				parentRoutine.symbolsByName.values(),
				implementSymbolAccessStringAndRemainder,
				iterators,
				parallelRegionTemplate,
				self._callee,
				useDeviceVersionIfAvailable=parentRoutine.implementation.onDevice
			)
		if not hasattr(self._callee, "implementation"):
			return arguments
		parallelRegionTemplate = None
		if isinstance(parentRegion, ParallelRegion):
			parallelRegionTemplate = parentRegion.template
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

	def clone(self):
		clone = super(CallRegion, self).clone()
		clone.loadCallee(self._callee)
		clone.loadPassedInSymbolsByName(self._passedInSymbolsByName)
		return clone

	def implemented(self, parentRoutine, parentRegion=None, skipDebugPrint=False):
		if not self._callee:
			raise Exception("call not loaded for call region in %s" %(parentRegion.name))

		text = ""
		calleeName = parentRoutine._adjustedCalleeNamesByName[self._callee.name]

		usedCompactedParameters = [
			s for s in sorted(parentRoutine._packedRealSymbolsByCalleeName.get(calleeName, []))
			if s.name in self._callee.usedSymbolNames
		]
		for idx, symbol in enumerate(usedCompactedParameters):
			text += "%s(%i) = %s" %(
				limitLength(frameworkArrayName(calleeName)),
				idx+1,
				symbol.nameInScope()
			) + " ! type %i symbol compaction for callee %s\n" %(symbol.declarationType, calleeName)

		parallelRegionPosition = None
		if hasattr(self._callee, "implementation"):
			parallelRegionPosition = self._callee.node.getAttribute("parallelRegionPosition")


		isForeignModuleCall = parentRoutine.parentModuleName != self._callee.parentModuleName
		if hasattr(self._callee, "implementation") and parallelRegionPosition == "within" and not isForeignModuleCall:
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
				calleeName,
				self._callee.implementation.kernelCallConfig()
			)
		else:
			text += "call " + calleeName

		text += "("
		if hasattr(self._callee, "implementation"):
			requiredAdditionalArgumentSymbols = [
				symbol for symbol in self._callee.additionalArgumentSymbols
				if symbol.name in self._callee.usedSymbolNames \
				and (not symbol.isTypeParameter or symbol.isDimensionParameter)
			]
			if len(requiredAdditionalArgumentSymbols) > 0:
				text += " &\n"
			bridgeStr1 = "&\n&"
			numOfProgrammerSpecifiedArguments = len(self._callee.programmerArguments)
			for symbolNum, symbol in enumerate(requiredAdditionalArgumentSymbols):
				symbolInCurrentContext = parentRoutine.symbolsByName.get(symbol.nameInScope(useDeviceVersionIfAvailable=False))
				if not symbolInCurrentContext:
					symbolInCurrentContext = parentRoutine.symbolsByName.get(symbol.name)
				if not symbolInCurrentContext:
					raise Exception("%s not found in context. All context keys: %s; Names in Scope: %s" %(
						symbol.nameInScope(useDeviceVersionIfAvailable=False),
						parentRoutine.symbolsByName.keys(),
						[s.nameInScope(useDeviceVersionIfAvailable=False) for s in parentRoutine.symbolsByName.values()]
				))
				hostName = symbolInCurrentContext.nameInScope()
				text += hostName
				if symbolNum < len(requiredAdditionalArgumentSymbols) - 1 or numOfProgrammerSpecifiedArguments > 0:
					text += ", %s" %(bridgeStr1)
		text += ", ".join(self._adjustedArguments(self._callee.programmerArguments, parentRoutine, parentRegion)) + ")\n"

		if hasattr(self._callee, "implementation") \
		and not self._callee.implementation.allowsMixedHostAndDeviceCode \
		and not isForeignModuleCall:
			activeSymbolsByName = dict(
				(symbol.name, symbol)
				for symbol in self._callee.additionalArgumentSymbols + self._passedInSymbolsByName.values()
				if symbol.name in self._callee.usedSymbolNamesInKernels
			)
			text += self._callee.implementation.kernelCallPost(
				activeSymbolsByName,
				self._callee.node
			)
		return self._sanitize(text, skipDebugPrint)

class ParallelRegion(Region):
	def __init__(self):
		self._currRegion = Region()
		self._subRegions = [self._currRegion]
		self._activeTemplate = None
		super(ParallelRegion, self).__init__()

	def __contains__(self, text):
		for region in self._subRegions:
			if text in region:
				return True
		return False

	@property
	def isCallingKernel(self):
		for region in self._subRegions:
			if region.isCallingKernel:
				return True
		return False

	@property
	def linesAndSymbols(self):
		return sum([
			region.linesAndSymbols
			for region in self._subRegions
		], [])

	@property
	def currRegion(self):
		return self._currRegion

	@property
	def template(self):
		return self._activeTemplate

	@property
	def usedSymbolNames(self):
		return set(sum([
			list(region.usedSymbolNames)
			for region in self._subRegions
		], []))

	def switchToRegion(self, region):
		self._currRegion = region
		self._subRegions.append(region)

	def loadLine(self, line, symbolsOnCurrentLine=None):
		self._currRegion.loadLine(line, symbolsOnCurrentLine)

	def loadActiveParallelRegionTemplate(self, templateNode):
		self._activeTemplate = templateNode

	def clone(self):
		clone = super(ParallelRegion, self).clone()
		if self._activeTemplate:
			clone.loadActiveParallelRegionTemplate(self._activeTemplate)
		clone._subRegions = []
		for region in self._subRegions:
			clonedRegion = region.clone()
			clone._subRegions.append(clonedRegion)
		clone._currRegion = clone._subRegions[0]
		return clone

	def firstAccessTypeOfScalar(self, symbol):
		for region in self._subRegions:
			accessType = region.firstAccessTypeOfScalar(symbol)
			if accessType != None:
				return accessType
		return None

	def implemented(self, parentRoutine, parentRegion=None, skipDebugPrint=False):
		text = ""
		hasAtExits = None
		routineHasKernels = parentRoutine.node.getAttribute('parallelRegionPosition') == 'within'
		if routineHasKernels and self._activeTemplate:
			text += parentRoutine.implementation.parallelRegionBegin(
				parentRoutine,
				[s for s in parentRoutine.symbolsByName.values() if s.name in self.usedSymbolNames],
				self._activeTemplate
			).strip() + "\n"
		else:
			hasAtExits = "@exit" in self
			if hasAtExits:
				text += parentRoutine.implementation.parallelRegionStubBegin()
		text += "\n".join([
			region.implemented(
				parentRoutine,
				parentRegion=self,
				skipDebugPrint=skipDebugPrint
			)
			for region in self._subRegions
		])
		if routineHasKernels and self._activeTemplate:
			text += parentRoutine.implementation.parallelRegionEnd(self._activeTemplate, parentRoutine).strip() + "\n"
		elif hasAtExits:
			text += parentRoutine.implementation.parallelRegionStubEnd()
		return self._sanitize(text, skipDebugPrint)

class RoutineSpecificationRegion(Region):
	def __init__(self):
		super(RoutineSpecificationRegion, self).__init__()
		self._additionalParametersByKernelName = None
		self._symbolsToAdd = None
		self._compactionDeclarationPrefixByCalleeName = None
		self._currAdditionalCompactedSubroutineParameters = None
		self._allImports = None
		self._typeParameterSymbolsByName = None
		self._dataSpecificationLines = []

	@property
	def usedSymbolNames(self):
		result = []
		for symbol in sum([symbols for _, symbols in self._linesAndSymbols], []):
			result += [tp.name for tp in symbol.usedTypeParameters if tp.isDimensionParameter]
		return set(result)

	def clone(self):
		clone = super(RoutineSpecificationRegion, self).clone()
		clone.loadAdditionalContext(
			self._additionalParametersByKernelName,
			self._symbolsToAdd,
			self._compactionDeclarationPrefixByCalleeName,
			self._currAdditionalCompactedSubroutineParameters,
			self._allImports
		)
		clone._dataSpecificationLines = copy.copy(self._dataSpecificationLines)
		clone._typeParameterSymbolsByName = copy.copy(self._typeParameterSymbolsByName)
		return clone

	def loadDataSpecificationLine(self, line):
		self._dataSpecificationLines.append(line)

	def loadAdditionalContext(
		self,
		additionalParametersByKernelName,
		symbolsToAdd,
		compactionDeclarationPrefixByCalleeName,
		currAdditionalCompactedSubroutineParameters,
		allImports
	):
		self._additionalParametersByKernelName = copy.copy(additionalParametersByKernelName)
		self._symbolsToAdd = copy.copy(symbolsToAdd)
		self._compactionDeclarationPrefixByCalleeName = copy.copy(compactionDeclarationPrefixByCalleeName)
		self._currAdditionalCompactedSubroutineParameters = copy.copy(currAdditionalCompactedSubroutineParameters)
		self._allImports = copy.copy(allImports)

	def loadTypeParameterSymbolsByName(self, typeParameterSymbolsByName):
		self._typeParameterSymbolsByName = copy.copy(typeParameterSymbolsByName)

	def firstAccessTypeOfScalar(self, symbol):
		return None

	def implemented(self, parentRoutine, parentRegion=None, skipDebugPrint=False):
		def getImportLine(importedSymbols, parentRoutine):
			return parentRoutine.implementation.getImportSpecification(
				importedSymbols,
				RegionType.KERNEL_CALLER_DECLARATION if parentRoutine.isCallingKernel else RegionType.OTHER,
				parentRoutine.node.getAttribute('parallelRegionPosition'),
				parentRoutine.parallelRegionTemplates
			)

		declarationRegionType = RegionType.OTHER
		if parentRoutine.isCallingKernel:
			declarationRegionType = RegionType.KERNEL_CALLER_DECLARATION

		if self._additionalParametersByKernelName == None \
		or self._symbolsToAdd == None \
		or self._compactionDeclarationPrefixByCalleeName == None \
		or self._currAdditionalCompactedSubroutineParameters == None:
			raise Exception("additional context not properly loaded for routine specification region in %s" %(
				parentRoutine.name
			))

		importsFound = False
		declaredSymbolsByScopedName = OrderedDict()
		textForKeywords = ""
		textBeforeDeclarations = ""
		textAfterDeclarations = ""
		declarations = ""
		symbolsToAddByScopedName = dict(
			(symbol.nameInScope(), symbol)
			for symbol in self._symbolsToAdd
		)
		iterators = set(getIterators(
			parentRoutine.node,
			parentRoutine.parallelRegionTemplates,
			parentRoutine.implementation.architecture
		))
		for (line, symbols) in self._linesAndSymbols:
			if not symbols or len(symbols) == 0:
				allImportMatch = regexPatterns.importAllPattern.match(line)
				selectiveImportMatch = regexPatterns.importPattern.match(line)
				if allImportMatch:
					importsFound = True
				elif selectiveImportMatch:
					importsFound = True
				elif not importsFound:
					textForKeywords += line.strip() + "\n"
				elif len(declaredSymbolsByScopedName.keys()) == 0:
					textBeforeDeclarations += line.strip() + "\n"
				else:
					textAfterDeclarations += line.strip() + "\n"
				continue
			for symbol in symbols:
				if symbol.nameInScope() in symbolsToAddByScopedName:
					continue
				if symbol.isCompacted:
					continue #compacted symbols are handled as part of symbolsToAdd
				if symbol.name in iterators:
					continue #will be added to declarations by implementation class
				specTuple = symbol.getSpecificationTuple(line)
				if specTuple[0]:
					declaredSymbolsByScopedName[symbol.nameInScope(useDeviceVersionIfAvailable=False)] = symbol
					symbol.loadDeclaration(
						specTuple,
						parentRoutine.programmerArguments,
						parentRoutine.name
					)
					continue
				match = symbol.importPattern.match(line)
				if not match:
					match = symbol.importMapPattern.match(line)
				if match:
					importsFound = True
					continue
				raise Exception("symbol %s expected to be referenced in line '%s', but all matchings have failed" %(
					symbol.name,
					line
				))

		text = ""
		if len(self._typeParameterSymbolsByName.keys()) > 0 \
		and conversionOptions.debugPrint \
		and not skipDebugPrint:
			text += "!<----- type parameters --\n"
		for typeParameterSymbol in self._typeParameterSymbolsByName.values():
			if typeParameterSymbol.sourceModule in parentRoutine.moduleNamesCompletelyImported:
				continue
			if typeParameterSymbol.isDimensionParameter:
				continue
			text += getImportLine([typeParameterSymbol], parentRoutine)
		if self._allImports:
			if len(self._allImports.keys()) > 0 and conversionOptions.debugPrint and not skipDebugPrint:
				text += "!<----- synthesized imports --\n"
			for (sourceModule, nameInScope) in self._allImports:
				if not nameInScope:
					text += getImportLine(sourceModule, parentRoutine)
					continue
				if sourceModule in parentRoutine.moduleNamesCompletelyImported:
					continue
				if nameInScope in self._typeParameterSymbolsByName \
				and not self._typeParameterSymbolsByName[nameInScope].isDimensionParameter:
					continue
				sourceName = self._allImports[(sourceModule, nameInScope)]
				symbol = parentRoutine.symbolsByName.get(sourceName)
				if symbol != None and symbol.sourceModule == parentRoutine.parentModuleName:
					continue
				if symbol != None:
					text += getImportLine([symbol], parentRoutine)
				else:
					adjustedSourceName = parentRoutine._adjustedCalleeNamesByName.get(sourceName, sourceName)
					adjustedNameInScope = parentRoutine._adjustedCalleeNamesByName.get(nameInScope, nameInScope)
					importSpecification = "use %s, only: %s => %s" %(sourceModule, adjustedNameInScope, adjustedSourceName) \
						if adjustedNameInScope != adjustedSourceName \
						else "use %s, only: %s" %(sourceModule, adjustedNameInScope)
					text += importSpecification
					if conversionOptions.debugPrint and not skipDebugPrint:
						text += " ! resynthesizing user input - no associated HF aware symbol found"
					text += "\n"

		if textForKeywords != "" and conversionOptions.debugPrint and not skipDebugPrint:
			text += "!<----- other imports and specs: ------\n"
		text += textForKeywords

		if textBeforeDeclarations != "" and conversionOptions.debugPrint and not skipDebugPrint:
			text += "!<----- before declarations: --\n"
		text += textBeforeDeclarations
		if len(declaredSymbolsByScopedName.keys()) > 0:
			if conversionOptions.debugPrint and not skipDebugPrint:
				text += "!<----- declarations: -------\n"
			declarations = "\n".join([
				parentRoutine.implementation.adjustDeclarationForDevice(
					symbol.getDeclarationLine(parentRoutine, purgeList=[]),
					[symbol],
					parentRoutine,
					declarationRegionType,
					parentRoutine.node.getAttribute('parallelRegionPosition')
				).strip()
				for symbol in declaredSymbolsByScopedName.values()
			]).strip() + "\n"
			text += declarations
		if len(self._dataSpecificationLines) > 0 and conversionOptions.debugPrint and not skipDebugPrint:
			text += "!<----- data specifications: --\n"
		if len(self._dataSpecificationLines) > 0:
			text += "\n".join(self._dataSpecificationLines) + "\n"
		if textAfterDeclarations != "" and conversionOptions.debugPrint and not skipDebugPrint:
			text += "!<----- after declarations: --\n"
		text += textAfterDeclarations

		#$$$ this needs to be adjusted for the unused symbols
		numberOfAdditionalDeclarations = (
			len(sum([
				self._additionalParametersByKernelName[kname][1]
				for kname in self._additionalParametersByKernelName
			], [])) + len(self._symbolsToAdd) + len(parentRoutine._packedRealSymbolsByCalleeName.keys())
		)

		if numberOfAdditionalDeclarations > 0 and conversionOptions.debugPrint and not skipDebugPrint:
			text += "!<----- auto emul symbols : --\n"
		defaultPurgeList = ['intent', 'public', 'parameter', 'allocatable', 'save']
		for symbol in self._symbolsToAdd:
			if not symbol.name in parentRoutine.usedSymbolNames:
				continue
			if symbol.isTypeParameter and not symbol.isDimensionParameter:
				continue
			if isinstance(symbol, FrameworkArray):
				symbol.compactedSymbols = [
					s for s in symbol.compactedSymbols
					if s.name in parentRoutine.usedSymbolNames
				]
				symbol.domains = [("hfauto", str(len(symbol.compactedSymbols)))]
			purgeList = defaultPurgeList
			if not symbol.isCompacted:
				purgeList=['public', 'parameter', 'allocatable', 'save']
			text += parentRoutine.implementation.adjustDeclarationForDevice(
				symbol.getDeclarationLine(parentRoutine, purgeList).strip(),
				[symbol],
				parentRoutine,
				declarationRegionType,
				parentRoutine.node.getAttribute('parallelRegionPosition')
			).rstrip() + " ! type %i symbol added for this subroutine\n" %(symbol.declarationType)
		for callee in parentRoutine.callees:
			#this hasattr is used to test the callee for analyzability without circular imports
			if not hasattr(callee, "implementation"):
				continue
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
				if not symbol.name in parentRoutine.usedSymbolNames:
					continue
				if symbol.isTypeParameter and not symbol.isDimensionParameter:
					continue
				if symbol.nameInScope(useDeviceVersionIfAvailable=False) in declaredSymbolsByScopedName:
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
					symbol.getDeclarationLine(parentRoutine, defaultPurgeList).strip(),
					[symbol],
					parentRoutine,
					declarationRegionType,
					parentRoutine.node.getAttribute('parallelRegionPosition')
				).rstrip() + " ! type %i symbol added for callee %s\n" %(symbol.declarationType, callee.name)
			toBeCompacted = [
				symbol for symbol in parentRoutine._packedRealSymbolsByCalleeName.get(callee.name, [])
				if symbol.name in callee.usedSymbolNames
			]
			if len(toBeCompacted) > 0:
				#TODO: generalize for cases where we don't want this to be on the device
				#(e.g. put this into Implementation class)
				compactedArray = FrameworkArray(
					callee.name,
					self._compactionDeclarationPrefixByCalleeName[callee.name],
					domains=[("hfauto", str(len(toBeCompacted)))],
					isOnDevice=True
				)
				text += implementation.adjustDeclarationForDevice(
					compactedArray.getDeclarationLine(parentRoutine).strip(),
					[compactedArray],
					parentRoutine,
					declarationRegionType,
					parentRoutine.node.getAttribute('parallelRegionPosition')
				).rstrip() + " ! compaction array added for callee %s\n" %(callee.name)

		declarationEndText = parentRoutine.implementation.declarationEnd(
			[
				s for s in parentRoutine.symbolsByName.values() + parentRoutine.additionalImports
				if s.isToBeTransfered or s.name in parentRoutine.usedSymbolNames
			],
			parentRoutine
		)
		if len(declarationEndText) > 0:
			text += "!<----- impl. specific decl end : --\n"
			text += declarationEndText

		usedCompactedParameters = [
			s for s in sorted(self._currAdditionalCompactedSubroutineParameters)
			if s.name in parentRoutine.usedSymbolNames
		]
		for idx, symbol in enumerate(usedCompactedParameters):
			text += "%s = %s(%i)" %(
				symbol.nameInScope(),
				limitLength(frameworkArrayName(parentRoutine.name)),
				idx+1
			) + " ! additional type %i symbol compaction\n" %(symbol.declarationType)

		return self._sanitize(text, skipDebugPrint)

class RoutineEarlyExitRegion(Region):
	def implemented(self, parentRoutine, parentRegion=None, skipDebugPrint=False):
		text = parentRoutine.implementation.subroutineExitPoint(
			[
				s for s in parentRoutine.symbolsByName.values()
				if s.isToBeTransfered or s.name in parentRoutine.usedSymbolNames
			],
			parentRoutine.isCallingKernel,
			isSubroutineEnd=False
		)
		text += super(RoutineEarlyExitRegion, self).implemented(
			parentRoutine,
			parentRegion=parentRegion,
			skipDebugPrint=True
		)
		return self._sanitize(text, skipDebugPrint)