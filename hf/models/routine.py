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

import copy, weakref, traceback
from models.region import RegionType, RoutineSpecificationRegion, ParallelRegion, CallRegion
from models.symbol import FrameworkArray, DeclarationType, ScopeError, limitLength, uniqueIdentifier
from machinery.commons import ConversionOptions, updateTypeParameterProperties
from tools.commons import UsageError

def getModuleArraysForCallee(calleeName, symbolAnalysisByRoutineNameAndSymbolName, symbolsByModuleNameAndSymbolName):
	moduleSymbols = []
	analysisBySymbolName = symbolAnalysisByRoutineNameAndSymbolName.get(calleeName, {})
	for symbolCallAnalysis in analysisBySymbolName.values():
		for symbolAnalysis in symbolCallAnalysis:
			if not symbolAnalysis.isModuleSymbol:
				continue
			symbol = symbolsByModuleNameAndSymbolName.get(symbolAnalysis.sourceModule, {}).get(symbolAnalysis.name)
			if symbol == None:
				#this happens for scalars for example
				continue
			symbol.analysis = symbolAnalysis
			moduleSymbols.append(symbol)
	return moduleSymbols

class Routine(object):
	def __init__(self, name, module, moduleRequiresStrongReference=False):
		if not type(name) in [str, unicode] or name.strip() == "":
			raise Exception("no valid name passed when trying to initialize routine")
		self.name = name
		self.isUsedInHostOnlyContext = False
		self._programmerArguments = None
		self._regions = None
		if moduleRequiresStrongReference:
			self._parentModule = module
		else:
			self._parentModule = weakref.ref(module)

	@property
	def regions(self):
		return self._regions

	@regions.setter
	def regions(self, _regions):
		self._regions = _regions

	@property
	def parentModule(self):
		if hasattr(self._parentModule, "createRoutine"):
			return self._parentModule
		return self._parentModule()

	@property
	def programmerArguments(self):
		if self._programmerArguments == None:
			raise Exception("programmer arguments not yet loaded for %s" %(self.name))
		return self._programmerArguments

	def loadArguments(self, arguments):
		self._programmerArguments = copy.copy(arguments)

	def nameInScope(self):
		return limitLength(self.name)

class AnalyzableRoutine(Routine):
	def __init__(self, name, module, routineNode, parallelRegionTemplates, implementation, moduleRequiresStrongReference=False):
		super(AnalyzableRoutine, self).__init__(name, module, moduleRequiresStrongReference)
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
		self.callees = []
		self._currRegion = RoutineSpecificationRegion(self)
		self._regions = [self._currRegion]
		self._additionalArguments = None
		self._additionalImports = None
		self._symbolsToUpdate = None
		self._moduleNodesByName = None
		self._symbolAnalysisByRoutineNameAndSymbolName = None
		self._symbolsByModuleNameAndSymbolName = None
		self._allImports = None
		self._userSpecifiedSymbolNames = None
		self._synthesizedSymbols = None
		self._packedRealSymbolsByCalleeName = {}
		self._adjustedCalleeNamesByName = None
		self.usedSymbolNames = {}
		self.usedSymbolNamesInKernels = {}
		self._moduleNamesCompletelyImported = None

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
	def isCallingKernel(self):
		if self.node.getAttribute("isKernelCaller") == "yes":
			return True
		for region in self._regions:
			if region.isCallingKernel:
				return True
		return False

	@property
	def moduleNamesCompletelyImported(self):
		if not self._allImports:
			return []
		if self._moduleNamesCompletelyImported != None:
			return self._moduleNamesCompletelyImported
		self._moduleNamesCompletelyImported = [
			sourceModule for (sourceModule, nameInScope) in self._allImports if nameInScope == None
		]
		return self._moduleNamesCompletelyImported

	def firstAccessTypeOfScalar(self, symbol):
		for region in self.regions:
			accessType = region.firstAccessTypeOfScalar(symbol)
			if accessType != None:
				return accessType
		return None

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
		for template in templates:
			if template == None:
				raise Exception("Invalid parallel region template loaded for %s" %(self.name))

	def _deduplicateAndFinalizeSymbols(self):
		def updateIndex(indexByNameAndScopeName, symbol):
			symbolsByScopeName = indexByNameAndScopeName.get(symbol.name, {})
			symbolsByScopeName[symbol.nameOfScope] = symbol
			indexByNameAndScopeName[symbol.name] = symbolsByScopeName

		def updateSymbolNode(symbol):
			if symbol.sourceModule == self.parentModule.name:
				symbol.routineNode = self.parentModule.node

		#scoped name could have changed through splitting / merging --> update symbolsByName
		symbolsByNameAndScopeName = {}
		for symbol in self.symbolsByName.values():
			updateIndex(symbolsByNameAndScopeName, symbol)
		parentModuleName = self.parentModule.name
		updatedSymbolsByName = {}
		for symbolsByScopeName in symbolsByNameAndScopeName.values():
			symbol = symbolsByScopeName.get(self.name)
			if not symbol:
				symbol = symbolsByScopeName.get(parentModuleName)
			if not symbol:
				symbol = symbolsByScopeName[symbolsByScopeName.keys()[0]] #$$$ this needs to be commented
				symbol.nameOfScope = self.name
			updateSymbolNode(symbol)
			if not isinstance(symbol, FrameworkArray):
				symbol.updateNameInScope(residingModule=self.parentModule.name)
			updatedSymbolsByName[symbol.nameInScope(useDeviceVersionIfAvailable=False)] = symbol.clone()
		for symbol in updatedSymbolsByName.values():
			for typeParameterSymbol in symbol.usedTypeParameters:
				if not self._userSpecifiedSymbolNames:
					raise Exception("%s has no _userSpecifiedSymbolNames" %(self.name))
				if typeParameterSymbol.name in self._userSpecifiedSymbolNames \
				and typeParameterSymbol.name in updatedSymbolsByName \
				and len(updatedSymbolsByName[typeParameterSymbol.name].domains) == 0 \
				and "integer" in updatedSymbolsByName[typeParameterSymbol.name].declarationPrefix:
					typeParameterSymbol.isUserSpecified = True
				typeParameterSymbol.nameOfScope = self.name
				updateSymbolNode(typeParameterSymbol)
				typeParameterSymbol.updateNameInScope(residingModule=self.parentModule.name)
			symbol.usedTypeParameters = set([typeParameter for typeParameter in symbol.usedTypeParameters])
		symbolsByOriginalName = dict(
			(symbol.name, symbol)
			for symbol in updatedSymbolsByName.values()
		)
		self.symbolsByName = updatedSymbolsByName

	def _updateSymbolState(self):
		regionType = RegionType.KERNEL_CALLER_DECLARATION if self.isCallingKernel else RegionType.OTHER
		updatedSymbolsByName = {}

		for symbol in self.symbolsByName.values():
			#reset symbol device state - could have been influenced by neighboring kernels
			symbol.isHostSymbol = False
			symbol.isPresent = False
			symbol.isToBeTransfered = False
			if symbol.attributes:
				symbol.setOptionsFromAttributes(symbol.attributes)
			for routine in [self] + self.callees:
				if not hasattr(routine, "usedSymbolNamesInKernels"):
					continue
				if symbol._isHostSymbol and symbol.name in routine.usedSymbolNamesInKernels:
					symbol.isPresent = True

			#some other symbol state resets
			symbol.parallelRegionPosition = self.node.getAttribute("parallelRegionPosition")
			symbol.isCompacted = False
			if not isinstance(symbol, FrameworkArray):
				symbol.loadRoutineNodeAttributes(self.node, self.parallelRegionTemplates)

			#implementation specific device state reset
			self.implementation.updateSymbolDeviceState(
				symbol,
				self.usedSymbolNamesInKernels,
				regionType,
				self.node.getAttribute("parallelRegionPosition")
			)

			#scoped naming state reset
			nameInScope = symbol.name
			if not isinstance(symbol, FrameworkArray):
				symbol.updateNameInScope(residingModule=self.parentModule.name)
			nameInScope = symbol.nameInScope(useDeviceVersionIfAvailable=False)
			updatedSymbolsByName[nameInScope] = symbol

		#compacted state reset
		for symbol in self.symbolsByName.values():
			if isinstance(symbol, FrameworkArray):
				for compactedSymbol in symbol.compactedSymbols:
					compactedSymbol.isCompacted = True

		self.symbolsByName = updatedSymbolsByName

	def _listCompactedSymbolsAndDeclarationPrefixAndOtherSymbols(self, additionalImports):
		toBeCompacted = []
		otherSymbols = []
		declarationPrefix = None
		purgeList=['intent', 'public', 'save', 'allocatable']
		for symbol in additionalImports:
			declType = symbol.declarationType

			#if this symbol is a parameter imported from a module (or local to our own), the defining element (e.g. '= 1.0d0')
			#needs to go away
			symbol.declarationSuffix = None

			# compact the imports of real type. Background: Experience has shown that too many
			# symbols passed to kernels, such that parameter list > 256Byte, can cause strange behavior. (corruption
			# of parameter list leading to launch failures)
			#-> need to pack all real symbols into an array to make it work reliably for cases where many reals are imported
			# why not integers? because they can be used as array boundaries.
			# Note: currently only a single real type per subroutine is supported for compaction
			#$$$ dangerous in case of mixed precition usage
			currentDeclarationPrefix = symbol.getSanitizedDeclarationPrefix(purgeList=purgeList)

			# CUDA Fortran supports parameters defined in kernels, at least as of v16.5
			if currentDeclarationPrefix \
			and declType in [
				DeclarationType.FOREIGN_MODULE_SCALAR,
				DeclarationType.LOCAL_MODULE_SCALAR,
				DeclarationType.OTHER_SCALAR,
				DeclarationType.LOCAL_SCALAR
			] \
			and not 'parameter' in currentDeclarationPrefix \
			and ( \
				'real' in currentDeclarationPrefix \
				or 'double' in currentDeclarationPrefix \
			) \
			and (declarationPrefix == None or currentDeclarationPrefix == declarationPrefix):
				declarationPrefix = currentDeclarationPrefix
				toBeCompacted.append(symbol)
			else:
				otherSymbols.append(symbol)
		arrayDeclarationPrefix = None
		if len(toBeCompacted) > 0:
			arrayDeclarationPrefix = toBeCompacted[0].getSanitizedDeclarationPrefix(purgeList=purgeList)
		return toBeCompacted, arrayDeclarationPrefix, otherSymbols

	def _prepareAdditionalContext(self):
		if not self._moduleNodesByName \
		or not self._symbolAnalysisByRoutineNameAndSymbolName \
		or not self._symbolsByModuleNameAndSymbolName:
			raise Exception("global context not loaded correctly for routine %s" %(self.name))

		try:
			#make sure that the symbols we have currently in scope have their attributes updated.
			#example where this was needed: local symbol that gets expanded with parallel dimensions otherwise
			#can get treated as compaction symbol for kernels where it isn't used as an argument
			for symbol in self.symbolsByName.values():
				symbol.parallelRegionPosition = self.node.getAttribute("parallelRegionPosition")
				symbol.loadRoutineNodeAttributes(self.node, self.parallelRegionTemplates)

			#build list of additional subroutine parameters
			#(parameters that the user didn't specify but that are necessary based on the features of the underlying technology
			# and the symbols declared by the user, such us temporary arrays and imported symbols)
			additionalImportsForOurSelves, \
			additionalDeclarationsForOurselves, \
			additionalDummiesForOurselves = self.implementation.getAdditionalKernelParameters(
				currRoutine=self,
				callee=self,
				moduleNodesByName=self._moduleNodesByName,
				symbolAnalysisByRoutineNameAndSymbolName=self._symbolAnalysisByRoutineNameAndSymbolName
			)

			symbolsByUniqueNameToBeUpdated = {}
			self._synthesizedSymbols = additionalImportsForOurSelves + additionalDeclarationsForOurselves + additionalDummiesForOurselves
			for symbol in self._synthesizedSymbols:
				symbolsByUniqueNameToBeUpdated[symbol.uniqueIdentifier] = symbol

			toBeCompacted, declarationPrefix, otherImports = self._listCompactedSymbolsAndDeclarationPrefixAndOtherSymbols(
				self._synthesizedSymbols
			)
			compactedArrayList = []
			additionalCompactedSubroutineParameters = []
			if len(toBeCompacted) > 0:
				compactedArray = FrameworkArray(
					self.name,
					declarationPrefix,
					domains=[("hfauto", str(len(toBeCompacted)))],
					isOnDevice=True
				)
				compactedArray.nameOfScope = self.name
				compactedArrayList = [compactedArray]
				additionalCompactedSubroutineParameters = sorted(toBeCompacted)
				compactedArray.compactedSymbols = additionalCompactedSubroutineParameters
			additionalSubroutineParameters = [s for s in sorted(otherImports + compactedArrayList) if s.declarationPrefix]
			self._additionalArguments = copy.copy(additionalSubroutineParameters)

			#analyse whether this routine is calling other routines that have a parallel region within
			#+ analyse the additional symbols that come up there
			additionalParametersByKernelName = {}
			additionalWrapperImportsByKernelName = {}
			if self.node.getAttribute("parallelRegionPosition") == "inside":
				for callee in self.callees:
					if not isinstance(callee, AnalyzableRoutine):
						continue
					if self.parentModule.name != callee.parentModule.name:
						continue

					#the callee routine has all the information about what's going on inside the routine,
					#, as opposed to only the interface information that is available in callee. this allows
					#detailed symbol usage analysis
					adjustedCalleeName = self._adjustedCalleeNamesByName.get(callee.name, callee.name)
					calleeRoutine = self.parentModule.routinesByName.get(adjustedCalleeName)
					if not calleeRoutine:
						calleeRoutine = callee

					additionalImportsForDeviceCompatibility, \
					additionalDeclarationsForDeviceCompatibility, \
					additionalDummies = callee.implementation.getAdditionalKernelParameters(
						currRoutine=self,
						callee=calleeRoutine,
						moduleNodesByName=self._moduleNodesByName,
						symbolAnalysisByRoutineNameAndSymbolName=self._symbolAnalysisByRoutineNameAndSymbolName
					)
					for symbol in additionalImportsForDeviceCompatibility \
						+ additionalDeclarationsForDeviceCompatibility \
						+ additionalDummies:
						self._synthesizedSymbols.append(symbol)
						symbolsByUniqueNameToBeUpdated[symbol.uniqueIdentifier] = symbol
					if 'DEBUG_PRINT' in callee.implementation.optionFlags:
						tentativeAdditionalImports = getModuleArraysForCallee(
							callee.name,
							self._symbolAnalysisByRoutineNameAndSymbolName,
							self._symbolsByModuleNameAndSymbolName
						)
						additionalImports = self.filterOutSymbolsAlreadyAliveInCurrentScope(tentativeAdditionalImports)
						additionalImportsByName = {}
						for symbol in additionalImports:
							additionalImportsByName[symbol.name] = symbol
						additionalWrapperImportsByKernelName[callee.name] = additionalImportsByName.values()
					additionalParametersByKernelName[callee.name] = (
						additionalImportsForDeviceCompatibility,
						additionalDeclarationsForDeviceCompatibility + additionalDummies
					)

			#prepare imports
			for symbolName in self.symbolsByName:
				symbol = self.symbolsByName[symbolName]
				if not symbol.uniqueIdentifier in symbolsByUniqueNameToBeUpdated:
					symbolsByUniqueNameToBeUpdated[symbol.uniqueIdentifier] = symbol
			self._symbolsToUpdate = symbolsByUniqueNameToBeUpdated.values()
			additionalImportsByScopedName = dict(
				(symbol.nameInScope(), symbol)
				for symbol in self.filterOutSymbolsAlreadyAliveInCurrentScope(
					sum(
						[
							additionalParametersByKernelName[kernelName][0]
							for kernelName in additionalParametersByKernelName.keys()
						],
						[]
					) + sum(
						[
							additionalWrapperImportsByKernelName[kernelName]
							for kernelName in additionalWrapperImportsByKernelName.keys()
						],
						[]
					)
				)
			)
			self._additionalImports = additionalImportsByScopedName.values()

			#finalize context for this routine
			ourSymbolsToAdd = additionalSubroutineParameters + additionalCompactedSubroutineParameters

			#prepare context in callees
			compactionDeclarationPrefixByCalleeName = {}
			for callee in self.callees:
				if not isinstance(callee, AnalyzableRoutine):
					continue
				additionalImports, additionalDeclarations = additionalParametersByKernelName.get(
					callee.name,
					([], [])
				)
				toBeCompacted, \
				declarationPrefix, \
				notToBeCompacted = self._listCompactedSymbolsAndDeclarationPrefixAndOtherSymbols(
					additionalImports + additionalDeclarations
				)
				toBeCompacted = sorted(toBeCompacted)
				if len(toBeCompacted) > 0:
					compactionDeclarationPrefixByCalleeName[callee.name] = declarationPrefix
					self._packedRealSymbolsByCalleeName[callee.name] = toBeCompacted
				compactedArrayList = []
				if len(toBeCompacted) > 0:
					compactedArray = FrameworkArray(
						callee.name,
						declarationPrefix,
						domains=[("hfauto", str(len(toBeCompacted)))],
						isOnDevice=True
					)
					compactedArray.nameOfScope = self.name
					compactedArrayList = [compactedArray]
					compactedArray.compactedSymbols = toBeCompacted
					self._synthesizedSymbols.append(compactedArray)
				callee._additionalArguments = copy.copy(sorted(notToBeCompacted + compactedArrayList))

			#load it into the specification region
			self.regions[0].loadAdditionalContext(
				additionalParametersByKernelName,
				ourSymbolsToAdd,
				compactionDeclarationPrefixByCalleeName,
				additionalCompactedSubroutineParameters,
				self._allImports
			)
		except Exception as e:
			raise Exception("In %s: %s;\nTraceback: %s" %(self.name, str(e), traceback.format_exc()))

	def _checkReferences(self, symbolList):
		for symbol in symbolList:
			contextSymbol = self.symbolsByName[
				symbol.nameInScope(useDeviceVersionIfAvailable=False)
			]
			if not symbol is contextSymbol:
				raise Exception("Symbol %s has an inconsistent context loaded in routine %s" %(
					symbol.name,
					self.name
				))

	def _mergeSynthesizedWithExistingSymbols(self):
		def updateReferences(inObject):
			symbolList = None
			outObject = None
			if isinstance(inObject, list):
				symbolList = inObject
				outObject = []
			elif isinstance(inObject, dict):
				symbolList = inObject.values()
				outObject = {}
			for symbol in symbolList:
				nameInScope = symbol.nameInScope(useDeviceVersionIfAvailable=False)
				matchedSymbol = self.symbolsByName.get(nameInScope)
				if not matchedSymbol:
					matchedSymbol = self.symbolsByName.get(symbol.name)
				if not matchedSymbol:
					matchedSymbol = self.symbolsByName.get(uniqueIdentifier(symbol.name, symbol.residingModule))
				if not matchedSymbol:
					matchedSymbol = self.symbolsByName.get(uniqueIdentifier(symbol.name, symbol.sourceModule))
				if not matchedSymbol:
					raise Exception("%s not found in context, cannot update references. All context keys: %s" %(
						nameInScope,
						self.symbolsByName.keys()
					))
				if isinstance(outObject, list):
					outObject.append(matchedSymbol)
				else:
					outObject[matchedSymbol.nameInScope(useDeviceVersionIfAvailable=False)] = matchedSymbol
			return outObject

		def updateLinesAndSymbols(region):
			updatedLinesAndSymbols = []
			for line, symbols in region._linesAndSymbols:
				updatedLinesAndSymbols.append((line, updateReferences(symbols)))
			region._linesAndSymbols = updatedLinesAndSymbols

		#update symbols in symbolsByName with additional ones
		for symbol in self._additionalArguments \
			+ self._synthesizedSymbols \
			+ self._symbolsToUpdate \
			+ self.regions[0]._symbolsToAdd:
			nameInScope = symbol.nameInScope(useDeviceVersionIfAvailable=False)
			if symbol.routineNode:
				symbol.updateNameInScope(residingModule=self.parentModule.name)
				nameInScope = symbol.nameInScope(useDeviceVersionIfAvailable=False)
			self.symbolsByName[nameInScope] = symbol

		#gather all the specified symbols
		for _, symbolsOnLine in self.regions[0].linesAndSymbols:
				for symbol in symbolsOnLine:
					self.symbolsByName[symbol.nameInScope(useDeviceVersionIfAvailable=False)] = symbol

		#make sure we only have one version of each symbol loaded. use overloading logic
		self._deduplicateAndFinalizeSymbols()

		#update symbols referenced on specific lines (could be replaced with automatically added ones)
		for region in self.regions:
			if hasattr(region, "_subRegions"):
				for subRegion in region._subRegions:
					updateLinesAndSymbols(subRegion)
			else:
				updateLinesAndSymbols(region)

		#update additional symbol lists
		self._additionalArguments = updateReferences(self._additionalArguments)
		self._synthesizedSymbols = updateReferences(self._synthesizedSymbols)
		self._symbolsToUpdate = updateReferences(self._symbolsToUpdate)
		self.regions[0]._symbolsToAdd = [
			s for s in updateReferences(self.regions[0]._symbolsToAdd)
			if s.declarationPrefix
		]
		for region in self.regions:
			if not isinstance(region, CallRegion):
				continue
			region._passedInSymbolsByName = updateReferences(region._passedInSymbolsByName)

		#gather type parameters (quadratic runtime!)
		allSymbolsInScope = self.symbolsByName.values()
		for symbol in allSymbolsInScope:
			updateTypeParameterProperties(symbol, allSymbolsInScope)

		#load type parameters into specification
		typeParametersByName = {}
		for symbol in allSymbolsInScope:
			for typeParameterSymbol in symbol.usedTypeParameters:
				if typeParameterSymbol.sourceModule == self.parentModule.name:
					continue
				typeParametersByName[typeParameterSymbol.name] = typeParameterSymbol
		self.regions[0].loadTypeParameterSymbolsByName(typeParametersByName)

	def _prepareCallRegions(self):
		def updateAdjustedCalleeNamesByName(region):
			adjustedCalleeName = self.implementation.adjustCalleeName(
				self,
				region._callee
			)
			self._adjustedCalleeNamesByName[region._callee.name] = adjustedCalleeName

		self._adjustedCalleeNamesByName = {}
		for region in self.regions:
			if isinstance(region, ParallelRegion):
				for subRegion in region._subRegions:
					if not isinstance(subRegion, CallRegion):
						continue
					subRegion.loadPassedInSymbolsByName(self.symbolsByName)
					updateAdjustedCalleeNamesByName(subRegion)
				continue
			if not isinstance(region, CallRegion):
				continue
			region.loadPassedInSymbolsByName(self.symbolsByName)
			updateAdjustedCalleeNamesByName(region)

	def _implementHeader(self):
		parameterList = ""
		requiredAdditionalArguments = [
			symbol for symbol in self._additionalArguments
			if symbol.name in self.usedSymbolNames \
			and (not symbol.isTypeParameter or symbol.isDimensionParameter)
		] if self._additionalArguments else []
		if requiredAdditionalArguments:
			parameterList += "&\n&"
			parameterList += "&, ".join([
				"%s & !additional type %i symbol inserted by framework \n" %(
					symbol.nameInScope(),
					symbol.declarationType
				)
				for symbol in requiredAdditionalArguments
			])
		if requiredAdditionalArguments and self._programmerArguments:
			parameterList += "&, "
		elif requiredAdditionalArguments:
			parameterList += "& "
		if self._programmerArguments:
			adjustedArguments = []
			for argument in self._programmerArguments:
				symbol = self.symbolsByName.get(argument)
				if symbol:
					adjustedArguments.append(symbol.nameInScope())
				else:
					adjustedArguments.append(argument)
			parameterList += ", ".join(adjustedArguments)
		return "%s subroutine %s(%s)\n" %(
			self.implementation.subroutinePrefix(self.node),
			self.name,
			parameterList
		)

	def _implementAdditionalImports(self):
		if not self._additionalImports or len(self._additionalImports) == 0:
			return self.implementation.additionalIncludes()
		return self.implementation.getImportSpecification(
			[i for i in self._additionalImports if not i.sourceModule in self.moduleNamesCompletelyImported],
			RegionType.KERNEL_CALLER_DECLARATION if self.isCallingKernel else RegionType.OTHER,
			self.node.getAttribute('parallelRegionPosition'),
			self.parallelRegionTemplates
		) + self.implementation.additionalIncludes()

	def _implementFooter(self):
		return self.implementation.subroutineExitPoint(
			[
				s for s in self.symbolsByName.values()
				if s.isToBeTransfered or s.name in self.usedSymbolNames
			],
			self.isCallingKernel,
			isSubroutineEnd=True
		) + "end subroutine\n"

	def _analyseSymbolUsage(self):
		for region in self._regions:
			for symbolName in region.usedSymbolNames:
				self.usedSymbolNames[symbolName] = None
				if isinstance(region, ParallelRegion):
					self.usedSymbolNamesInKernels[symbolName] = None
		if self._additionalArguments:
			for symbol in self._additionalArguments:
				if not isinstance(symbol, FrameworkArray):
					continue
				for compactedSymbol in symbol.compactedSymbols:
					if compactedSymbol.name in self.usedSymbolNames:
						self.usedSymbolNames[symbol.name] = None
		for symbolName in copy.copy(self.usedSymbolNames.keys()):
			usedSymbol = self.symbolsByName.get(symbolName)
			if not usedSymbol:
				continue
			for typeParameter in usedSymbol.usedTypeParameters:
				if not typeParameter.isDimensionParameter:
					continue #non-dimension typeparameters are handled separately in spec. region
				self.usedSymbolNames[typeParameter.name] = None

		#$$$ this is a quick fix for missing scope, because some type parameters
		# don't get loaded correctly. Check adv_calcflxz_1d as a testcase.
		symbolsByUserDefinedName = {}
		for symbol in self.symbolsByName.values():
			symbolsByUserDefinedName[symbol.name] = symbol
		for symbolName in self._userSpecifiedSymbolNames:
			symbol = symbolsByUserDefinedName.get(symbolName)
			if not symbol:
				continue
			if symbol.domains:
				continue #only add scalars
			if symbol.declarationType == DeclarationType.LOCAL_MODULE_SCALAR \
			and not symbol.isDimensionParameter:
				continue #don't include general module scalars here - programmer has to add those in each routine manually
			if not "integer" in symbol.declarationPrefix:
				continue #we only want dimension parameters; note: adv_calcflxz_1d doesn't recognise some of them as dimension params, so this is a workaround.
			self.usedSymbolNames[symbol.name] = None

	def checkSymbols(self):
		self._checkReferences(self._additionalArguments)
		self._checkReferences(self._synthesizedSymbols)
		self._checkReferences(self._symbolsToUpdate)
		self._checkReferences(self.regions[0]._symbolsToAdd)
		for s in self.regions[0]._symbolsToAdd:
			if not s.declarationPrefix:
				raise ScopeError("invalid symbol %s added to %s" %(s.name, self.name))
		for region in self.regions:
			self._checkReferences(sum([las[1] for las in region.linesAndSymbols], []))

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

	#creates a clone with meta data but no region data
	def createCloneWithMetadata(self, name):
		clone = AnalyzableRoutine(
			name,
			self.parentModule,
			routineNode=self.node.cloneNode(deep=True),
			parallelRegionTemplates=copy.copy(self.parallelRegionTemplates),
			implementation=self.implementation
		)
		clone._programmerArguments = copy.copy(self._programmerArguments)
		clone._additionalArguments = copy.copy(self._additionalArguments)
		clone._additionalImports = copy.copy(self._additionalImports)
		clone._symbolsToUpdate = copy.copy(self._symbolsToUpdate)
		clone._moduleNodesByName = self._moduleNodesByName
		clone._symbolAnalysisByRoutineNameAndSymbolName = self._symbolAnalysisByRoutineNameAndSymbolName
		clone._symbolsByModuleNameAndSymbolName = self._symbolsByModuleNameAndSymbolName
		clone._allImports = copy.copy(self._allImports)
		clone._userSpecifiedSymbolNames = copy.copy(self._userSpecifiedSymbolNames)
		clone.symbolsByName = copy.copy(self.symbolsByName)
		clone.callees = copy.copy(self.callees)
		clone.isUsedInHostOnlyContext = self.isUsedInHostOnlyContext
		return clone

	def clone(self, cloneName):
		clone = self.createCloneWithMetadata(cloneName)
		clone.resetRegions([region.clone() for region in self._regions])
		return clone

	def resetRegions(self, regions):
		self._regions = []
		if isinstance(regions, list):
			for entry in regions:
				self.addRegion(entry)
		else:
			self.addRegion(regions)

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

	def loadSymbolsByName(self, symbolsByName):
		self.symbolsByName = copy.copy(symbolsByName)

	def loadAllImports(self, allImports):
		self._allImports = copy.copy(allImports)

	def loadGlobalContext(
		self,
		moduleNodesByName,
		symbolAnalysisByRoutineNameAndSymbolName,
		symbolsByModuleNameAndSymbolName
	):
		self._moduleNodesByName = moduleNodesByName
		self._symbolAnalysisByRoutineNameAndSymbolName = symbolAnalysisByRoutineNameAndSymbolName
		self._symbolsByModuleNameAndSymbolName = symbolsByModuleNameAndSymbolName

	def loadCall(self, callRoutine, overrideRegion=None):
		callRegion = None
		if overrideRegion != None:
			callRegion = overrideRegion
		elif isinstance(self._currRegion, CallRegion):
			callRegion = self._currRegion
		elif isinstance(self._currRegion, ParallelRegion) \
		and isinstance(self._currRegion.currRegion, CallRegion):
			callRegion = self._currRegion.currRegion
		if not isinstance(callRegion, CallRegion):
			raise Exception("cannot load call %s for %s outside a callregion" %(
				callRoutine.name,
				self.name
			))
		callRegion.loadCallee(callRoutine)
		self.callees.append(callRoutine)

	def loadLine(self, line, symbolsOnCurrentLine=None):
		self._currRegion.loadLine(line, symbolsOnCurrentLine)

	def finalize(self):
		self._userSpecifiedSymbolNames = dict(
			(symbol.name, None)
			for symbol in self.symbolsByName.values()
			if symbol.isUserSpecified
		)

	def implemented(self):
		purgedRoutineElements = []
		try:
			self._updateSymbolState()
			self.checkSymbols()
			implementedRoutineElements = [self._implementHeader(), self._implementAdditionalImports()]
			implementedRoutineElements += [region.implemented() for region in self._regions]
			implementedRoutineElements += [self._implementFooter()]
			purgedRoutineElements = [
				(index, text) for index, text in enumerate(implementedRoutineElements)
				if text != ""
			]
		except UsageError as e:
			raise UsageError("In %s: %s" %(self.name, str(e)))
		except ScopeError as e:
			raise ScopeError("In %s: %s;\nTraceback: %s" %(self.name, str(e), traceback.format_exc()))
		except Exception as e:
			raise ScopeError("In %s: %s;\nTraceback: %s" %(self.name, str(e), traceback.format_exc()))
		return "\n".join([
			text
			for (index, text) in purgedRoutineElements
		])
