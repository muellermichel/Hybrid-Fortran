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
from models.symbol import FrameworkArray, DeclarationType, ScopeError, limitLength
from machinery.commons import ConversionOptions
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

def uniqueIdentifier(routineName, implementationName):
	return (routineName + "_hfauto_" + implementationName).strip()

class Routine(object):
	def __init__(self, name, module):
		if not type(name) in [str, unicode] or name.strip() == "":
			raise Exception("no valid name passed when trying to initialize routine")
		self.name = name
		self._programmerArguments = None
		self._parentModule = weakref.ref(module)

	@property
	def parentModule(self):
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
	def __init__(self, name, module, routineNode, parallelRegionTemplates, implementation):
		super(AnalyzableRoutine, self).__init__(name, module)
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
		self._packedRealSymbolsByCalleeName = {}

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
			and isinstance(region._callee, AnalyzableRoutine) \
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

	def _updateSymbolReferences(self):
		#scoped name could have changed through splitting / merging
		for symbol in self.symbolsByName.values():
			symbol.resetScope(self.name)
		symbolsByNameAndScopeName = {}
		for symbol in self.symbolsByName.values():
			symbolsByScopeName = symbolsByNameAndScopeName.get(symbol.name, {})
			symbolsByScopeName[symbol.nameOfScope] = symbol
			symbolsByNameAndScopeName[symbol.name] = symbolsByScopeName
		parentModuleName = self._parentModule().name
		updatedSymbolsByName = {}
		for symbolsByScopeName in symbolsByNameAndScopeName.values():
			symbol = symbolsByScopeName.get(self.name)
			if not symbol:
				symbol = symbolsByScopeName.get(parentModuleName)
			if not symbol:
				symbol = symbolsByScopeName[symbolsByScopeName.keys()[0]]
				symbol.nameOfScope = self.name
			symbol.updateNameInScope()
			updatedSymbolsByName[symbol.nameInScope(useDeviceVersionIfAvailable=False)] = symbol
		self.symbolsByName = updatedSymbolsByName

	def _updateSymbolState(self):
		#updating device state
		if self._symbolsToUpdate == None:
			raise Exception("no symbols loaded for updating in routine %s" %(self.name))
		regionType = RegionType.KERNEL_CALLER_DECLARATION if self.isCallingKernel else RegionType.OTHER
		for symbol in self._symbolsToUpdate:
			symbol.parallelRegionPosition = self.node.getAttribute("parallelRegionPosition")
			self.implementation.updateSymbolDeviceState(
				symbol,
				regionType,
				self.node.getAttribute("parallelRegionPosition")
			)

	def _loadAdditionalArgumentSymbols(self, additionalArgumentSymbols):
		self._additionalArguments = copy.copy(additionalArgumentSymbols)

	def _listCompactedSymbolsAndDeclarationPrefixAndOtherSymbols(self, additionalImports):
		toBeCompacted = []
		otherImports = []
		declarationPrefix = None
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
			currentDeclarationPrefix = symbol.getSanitizedDeclarationPrefix()
			if declType in [
				DeclarationType.FOREIGN_MODULE_SCALAR,
				DeclarationType.LOCAL_MODULE_SCALAR,
				DeclarationType.OTHER_SCALAR,
				DeclarationType.LOCAL_SCALAR
			] \
			and ( \
				'real' in currentDeclarationPrefix \
				or 'double' in currentDeclarationPrefix \
			) \
			and (declarationPrefix == None or currentDeclarationPrefix == declarationPrefix):
				declarationPrefix = currentDeclarationPrefix
				symbol.isCompacted = True
				toBeCompacted.append(symbol)
			else:
				otherImports.append(symbol)
		return toBeCompacted, declarationPrefix, otherImports

	def _prepareAdditionalContext(self):
		if not self._moduleNodesByName \
		or not self._symbolAnalysisByRoutineNameAndSymbolName \
		or not self._symbolsByModuleNameAndSymbolName:
			raise Exception("global context not loaded correctly for routine %s" %(self.name))

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
		for symbol in additionalImportsForOurSelves + additionalDeclarationsForOurselves:
			symbol.isEmulatingSymbolThatWasActiveInCurrentScope = True

		symbolsByUniqueNameToBeUpdated = {}
		additionalParameters = additionalImportsForOurSelves + additionalDeclarationsForOurselves + additionalDummiesForOurselves
		for symbol in additionalParameters:
			symbolsByUniqueNameToBeUpdated[symbol.uniqueIdentifier] = symbol
			self.symbolsByName[symbol.uniqueIdentifier] = symbol

		toBeCompacted, declarationPrefix, otherImports = self._listCompactedSymbolsAndDeclarationPrefixAndOtherSymbols(
			additionalParameters
		)
		compactedArrayList = []
		if len(toBeCompacted) > 0:
			compactedArray = FrameworkArray(
				self.name,
				declarationPrefix,
				domains=[("hfauto", str(len(toBeCompacted)))],
				isOnDevice=True
			)
			compactedArrayList = [compactedArray]
		additionalSubroutineParameters = sorted(otherImports + compactedArrayList)
		self._loadAdditionalArgumentSymbols(additionalSubroutineParameters)

		#analyse whether this routine is calling other routines that have a parallel region within
		#+ analyse the additional symbols that come up there
		additionalParametersByKernelName = {}
		additionalWrapperImportsByKernelName = {}
		if self.node.getAttribute("parallelRegionPosition") == "inside":
			for callee in self.callees:
				if not isinstance(callee, AnalyzableRoutine):
					continue
				additionalImportsForDeviceCompatibility, \
				additionalDeclarationsForDeviceCompatibility, \
				additionalDummies = callee.implementation.getAdditionalKernelParameters(
					currRoutine=self,
					callee=callee,
					moduleNodesByName=self._moduleNodesByName,
					symbolAnalysisByRoutineNameAndSymbolName=self._symbolAnalysisByRoutineNameAndSymbolName
				)
				for symbol in additionalImportsForDeviceCompatibility \
				+ additionalDeclarationsForDeviceCompatibility \
				+ additionalDummies:
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
						symbol.resetScope(self.name)
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
		additionalCompactedSubroutineParameters = sorted(toBeCompacted)
		ourSymbolsToAdd = sorted(
			additionalSubroutineParameters + additionalCompactedSubroutineParameters
		)

		#prepare context in callees and load it into our specification region
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
				compactedArrayList = [compactedArray]
			callee._loadAdditionalArgumentSymbols(sorted(notToBeCompacted + compactedArrayList))

		#load into the specification region
		self.regions[0].loadAdditionalContext(
			additionalParametersByKernelName,
			ourSymbolsToAdd,
			compactionDeclarationPrefixByCalleeName,
			additionalCompactedSubroutineParameters,
			self._allImports
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
		return self.implementation.getImportSpecification(
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
			self._parentModule(),
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
		clone.symbolsByName = copy.copy(self.symbolsByName)
		clone.callees = copy.copy(self.callees)
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

	def implemented(self):
		purgedRoutineElements = []
		try:
			self._checkParallelRegions()
			self._prepareAdditionalContext()
			self._updateSymbolReferences()
			self._updateSymbolState()
			implementedRoutineElements = [self._implementHeader(), self._implementAdditionalImports()]
			implementedRoutineElements += [region.implemented() for region in self._regions]
			implementedRoutineElements += [self._implementFooter()]
			purgedRoutineElements = [
				(index, text) for index, text in enumerate(implementedRoutineElements)
				if text != ""
			]
		except UsageError as e:
			raise UsageError("Error in %s: %s" %(self.name, str(e)))
		except ScopeError as e:
			raise ScopeError("Error in %s: %s;\nTraceback: %s" %(self.name, str(e), traceback.format_exc()))
		return "\n".join([
			text
			for (index, text) in purgedRoutineElements
		])
