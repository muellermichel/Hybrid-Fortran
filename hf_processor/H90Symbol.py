#!/usr/bin/python
# -*- coding: UTF-8 -*-

# Copyright (C) 2013 Michel Müller (Typhoon Computing), RIKEN Advanced Institute for Computational Science (AICS)

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

#**********************************************************************#
#  Procedure        FortranImplementation.py                           #
#  Comment          Put together valid fortran strings according to    #
#                   xml data and parser data                           #
#  Date             2012/08/02                                         #
#  Author           Michel Müller (AOKI Laboratory)                    #
#**********************************************************************#

import re, sys, copy
import pdb
from DomHelper import *
from GeneralHelper import enum, BracketAnalyzer

Init = enum("NOTHING_LOADED",
	"DEPENDANT_ENTRYNODE_ATTRIBUTES_LOADED",
	"ROUTINENODE_ATTRIBUTES_LOADED",
	"DECLARATION_LOADED"
)

#Set of declaration types that are mutually exlusive for
#declaration lines in Hybrid Fortran.
#-> You cannot mix and match declarations of different types
DeclarationType = enum("UNDEFINED",
	"LOCAL_ARRAY",
	"IMPORTED_SCALAR",
	"MODULE_SCALAR",
	"FRAMEWORK_ARRAY",
	"OTHER"
)

def splitDeclarationSettingsFromSymbols(line, dependantSymbols, patterns, withAndWithoutIntent=True):
	declarationDirectives = ""
	symbolDeclarationStr = ""
	if patterns.symbolDeclTestPattern.match(line):
		match = patterns.symbolDeclPattern.match(line)
		if not match:
			raise Exception("When trying to extract a device declaration: This is not a valid declaration: %s" %(line))
		declarationDirectives = match.group(1)
		symbolDeclarationStr = match.group(2)
	else:
		#no :: is used in this declaration line -> we should only have one symbol defined on this line
		if len(dependantSymbols) > 1:
			raise Exception("Declaration line without :: has multiple matching dependant symbols.")
		match = re.match(r"(\s*(?:double\s+precision|real|integer|character|logical)(?:.*?))\s*(" + re.escape(dependantSymbols[0].name) + r".*)", line, re.IGNORECASE)
		if not match:
			raise Exception("When trying to extract a device declaration: This is not a valid declaration: %s" %(line))
		declarationDirectives = match.group(1)
		symbolDeclarationStr = match.group(2)

	if not withAndWithoutIntent:
		return declarationDirectives, symbolDeclarationStr

	declarationDirectivesWithoutIntent = ""
	match = re.match(r"(.*?),\s*intent.*?\)(.*)", declarationDirectives, re.IGNORECASE)
	if match:
		declarationDirectivesWithoutIntent = match.group(1) + match.group(2)
	else:
		declarationDirectivesWithoutIntent = declarationDirectives

	return declarationDirectivesWithoutIntent, declarationDirectives, symbolDeclarationStr

def purgeDimensionAndGetAdjustedLine(line, patterns):
	match = patterns.dimensionPattern.match(line)
	if not match:
		return line
	else:
		return match.group(1) + match.group(3)

class Symbol(object):
	name = None
	intent = None
	domains = []
	template = None
	isMatched = False
	isOnDevice = False
	isUsingDevicePostfix = False
	isPresent = False
	isAutomatic = False
	isPointer = False
	isAutoDom = False
	isToBeTransfered = False
	isCompacted = False
	declPattern = None
	namePattern = None
	importPattern = None
	pointerAssignmentPattern = None
	parallelRegionPosition = None
	numOfParallelDomains = 0
	parallelActiveDims = [] #!Important: The order of this list must remain insignificant when it is used
	parallelInactiveDims = [] #!Important: The order of this list must remain insignificant when it is used
	aggregatedRegionDomSizesByName = {}
	routineNode = None
	declarationPrefix = None
	initLevel = Init.NOTHING_LOADED
	sourceModule = None
	sourceSymbol = None
	debugPrint = None

	def __init__(self, name, template, isAutomatic=False, debugPrint=False):
		if not name or name == "":
			raise Exception("Unexpected error: name required for initializing symbol")
		if template == None:
			raise Exception("Unexpected error: template required for initializing symbol")

		self.name = name
		self.template = template
		self.isAutomatic = isAutomatic
		self.isPointer = False
		self.debugPrint = debugPrint
		self.domains = []
		self.isMatched = False
		self.declPattern = re.compile(r'(\s*(?:double\s+precision|real|integer|logical).*?[\s,:]+)' + re.escape(name) + r'((?:\s|\,|\(|$)+.*)', \
			re.IGNORECASE)
		self.namePattern = re.compile(r'((?:[^\"\']|(?:\".*\")|(?:\'.*\'))*?(?:\W|^))' + re.escape(name) + r'(?:_d)?((?:\W.*)|\Z)', \
			re.IGNORECASE)
		self.symbolImportPattern = re.compile(r'^\s*use\s*(\w*)[,\s]*only\s*\:.*?\W' + re.escape(name) + r'\W.*', \
			re.IGNORECASE)
		self.symbolImportMapPattern = re.compile(r'.*?\W' + re.escape(name) + r'\s*\=\>\s*(\w*).*', \
			re.IGNORECASE)
		self.pointerDeclarationPattern = re.compile(r'\s*(?:double\s+precision|real|integer|logical).*?pointer.*?[\s,:]+' + re.escape(name), re.IGNORECASE)
		self.parallelRegionPosition = None
		self.isUsingDevicePostfix = False
		self.isOnDevice = False
		self.parallelActiveDims = []
		self.parallelInactiveDims = []
		self.aggregatedRegionDomSizesByName = {}
		self.aggregatedRegionDomNames = []
		self.routineNode = None
		self.declarationPrefix = None
		self.initLevel = Init.NOTHING_LOADED
		self.sourceModule = None
		self.sourceSymbol = None

		self.isPresent = False
		self.isAutoDom = False
		self.isToBeTransfered = False
		self.isCompacted = False
		attributes = getAttributes(self.template)
		if "present" in attributes:
			self.isPresent = True
		if "autoDom" in attributes:
			self.isAutoDom = True
		if "transferHere" in attributes:
			if self.isPresent:
				raise Exception("Symbol %s has contradicting attributes 'transferHere' and 'present'" %(self))
			self.isToBeTransfered = True

	def __repr__(self):
		name = self.name
		if self.isAutomatic:
			name = self.automaticName()
		elif len(self.domains) > 0:
			name = self.deviceName()
		if len(self.domains) == 0:
			return name

		try:
			result = name
			domPP = self.domPP()
			if self.parallelRegionPosition != "outside" and domPP != "":
				result = result + "(" + domPP + "("
			else:
				result = result + "("
			for i in range(len(self.domains)):
				if i != 0:
					result += ", "
				if self.isPointer:
					result += ":"
				else:
					(domName, domSize) = self.domains[i]
					result += domSize
			if self.parallelRegionPosition != "outside" and domPP != "":
				result = result + "))"
			else:
				result = result + ")"
		except Exception:
			return "%s{%s}" %(name, str(self.domains))
		return result

	def __eq__(self, other):
		return self.automaticName() == other.automaticName()
	def __ne__(self, other):
		return self.automaticName() != other.automaticName()
	def __lt__(self, other):
		return self.automaticName() < other.automaticName()
	def __le__(self, other):
		return self.automaticName() <= other.automaticName()
	def __gt__(self, other):
		return self.automaticName() > other.automaticName()
	def __ge__(self, other):
		return self.automaticName() >= other.automaticName()

	@property
	def numOfParallelDomains(self):
		if self.parallelRegionPosition == "outside":
			return 0
		return len(self.parallelActiveDims)

	def storeDomainDependantEntryNodeAttributes(self, domainDependantEntryNode):
		if self.intent:
			domainDependantEntryNode.setAttribute("intent", self.intent)
		if self.declarationPrefix:
			domainDependantEntryNode.setAttribute("declarationPrefix", self.declarationPrefix)
		if self.sourceModule:
			domainDependantEntryNode.setAttribute("sourceModule", self.sourceModule)
		if self.sourceSymbol:
			domainDependantEntryNode.setAttribute("sourceSymbol", self.sourceSymbol)
		domainDependantEntryNode.setAttribute("isPointer", "yes" if self.isPointer else "no")
		if self.domains and len(self.domains) > 0:
			domainDependantEntryNode.setAttribute(
				"declaredDimensionSizes", ",".join(
					[dimSize for _, dimSize in self.domains]
				)
			)

	def loadDomainDependantEntryNodeAttributes(self, domainDependantEntryNode):
		if self.initLevel > Init.NOTHING_LOADED:
			sys.stderr.write("WARNING: symbol %s's entry node attributes are loaded when the initialization level has already advanced further\n" \
				%(str(self))
			)

		self.intent = domainDependantEntryNode.getAttribute("intent")
		self.declarationPrefix = domainDependantEntryNode.getAttribute("declarationPrefix")
		self.sourceModule = domainDependantEntryNode.getAttribute("sourceModule")
		self.sourceSymbol = domainDependantEntryNode.getAttribute("sourceSymbol")
		self.isPointer = domainDependantEntryNode.getAttribute("isPointer") == "yes"
		for dimSize in domainDependantEntryNode.getAttribute("declaredDimensionSizes").split(","):
			if dimSize.strip() != "":
				self.domains.append(('HF_GENERIC_DIM', dimSize))
		self.initLevel = max(self.initLevel, Init.DEPENDANT_ENTRYNODE_ATTRIBUTES_LOADED)

	def checkIntegrityOfDomains(self):
		for domain in self.domains:
			if not hasattr(domain, '__iter__'):
				raise Exception("Invalid definition of domain in symbol %s: %s" %(self.name, str(domain)))

	def loadRoutineNodeAttributes(self, routineNode, parallelRegionTemplates):
		if self.initLevel < Init.DEPENDANT_ENTRYNODE_ATTRIBUTES_LOADED:
			raise Exception("Symbol %s's routine node attributes are loaded without loading the entry node attributes first."
				%(str(self))
			)
		if self.initLevel > Init.DEPENDANT_ENTRYNODE_ATTRIBUTES_LOADED:
			sys.stderr.write("WARNING: symbol %s's routine node attributes are loaded when the initialization level has already advanced further\n" \
				%(str(self))
			)

		self.routineNode = routineNode

		#get and check parallelRegionPosition
		routineName = routineNode.getAttribute("name")
		if not routineName:
			raise Exception("Unexpected error: routine node without name: %s" %(routineNode.toxml()))
		parallelRegionPosition = routineNode.getAttribute("parallelRegionPosition")

		if not parallelRegionPosition or parallelRegionPosition == "":
			return #when this routine is called in the declaraton extractor script (stage 1) -> parallel regions not analyzed yet.

		if parallelRegionPosition not in ["inside", "outside", "within"]:
			raise Exception("Invalid parallel region position definition ('%s') for routine %s" %(parallelRegionPosition, routineName))
		self.parallelRegionPosition = parallelRegionPosition

		dependantDomNameAndSize = getDomNameAndSize(self.template)
		dependantDomSizeByName = {
			dependantDomName:dependantDomSize
			for (dependantDomName, dependantDomSize)
			in dependantDomNameAndSize
		}
		declarationPrefixFromTemplate = getDeclarationPrefix(self.template)
		if declarationPrefixFromTemplate != None and declarationPrefixFromTemplate.strip() != "":
			self.declarationPrefix = declarationPrefixFromTemplate
		#   which of those dimensions are invariants in               #
		#   the currently active parallel regions?                    #
		#   -> put them in the 'parallelActive' set, put the          #
		#   others in the 'parallelInactive' set.                     #
		if not parallelRegionTemplates or len(parallelRegionTemplates) == 0:
			sys.stderr.write("WARNING: No active parallel region found, in subroutine %s where dependants are defined\n" %(routineName))
			return
		if len(parallelRegionTemplates) > 1 and parallelRegionPosition != "inside":
			raise Exception("Only one active parallel region definition allowed within a subroutines or in an outside callgraph position. %i found for %s" \
				%(len(parallelRegionTemplates), self.currSubprocName))
		self.parallelActiveDims = []
		self.parallelInactiveDims = []
		self.aggregatedRegionDomNames = []
		self.aggregatedRegionDomSizesByName = {}
		for parallelRegionTemplate in parallelRegionTemplates:
			regionDomNameAndSize = getDomNameAndSize(parallelRegionTemplate)
			for (regionDomName, regionDomSize) in regionDomNameAndSize:
				if regionDomName in dependantDomSizeByName and regionDomName not in self.parallelActiveDims:
					self.parallelActiveDims.append(regionDomName)
				self.aggregatedRegionDomSizesByName[regionDomName] = regionDomSize
				self.aggregatedRegionDomNames.append(regionDomName)

		for (dependantDomName, dependantDomSize) in dependantDomNameAndSize:
			if dependantDomName not in self.parallelActiveDims:
				self.parallelInactiveDims.append(dependantDomName)

		dimsBeforeReset = self.domains
		self.domains = []
		for (dependantDomName, dependantDomSize) in dependantDomNameAndSize:
			if dependantDomName not in self.parallelActiveDims and \
			dependantDomName not in self.parallelInactiveDims:
				raise Exception("Automatic symbol %s's dependant domain size %s is not declared as one of its dimensions." \
					%(self.name, dependantDomSize))
			self.domains.append((dependantDomName, dependantDomSize))
		if self.isAutoDom and not self.isPointer:
			for dim in dimsBeforeReset:
				self.domains.append(dim)

		self.checkIntegrityOfDomains()
		self.initLevel = max(self.initLevel, Init.ROUTINENODE_ATTRIBUTES_LOADED)
		if self.debugPrint:
			sys.stderr.write("routine node attributes loaded for symbol %s. Domains at this point: %s\n" %(self.name, str(self.domains)))

	def loadDeclaration(self, paramDeclMatch, patterns):
		def getNextUnusedIndexForDimensionSize(domainSize, dimensionSizesInDeclaration, usedIndices):
			index_candidate = None
			startAt = 0
			while True:
				if startAt > len(dimensionSizesInDeclaration) - 1:
					raise Exception("domain size %s cannot be matched in symbol %s's declaration." %(str(domainSize), symbol.name))
				try:
					index_candidate = dimensionSizesInDeclaration[startAt:].index(domainSize) + startAt
				except ValueError:
					raise Exception("Dimension Size '%s' expected in symbol %s's declaration - not found." %(domainSize, self.name))
				if index_candidate in usedIndices:
					startAt = index_candidate + 1
				else:
					break;
			return index_candidate


		def getReorderedDomainsAccordingToDeclaration(domains, dimensionSizesInDeclaration):
			if len(domains) != len(dimensionSizesInDeclaration) or len(domains) == 0:
				return domains
			if self.debugPrint:
				sys.stderr.write("reordering domains (%s) for symbol %s according to declaration (%s).\n"
					%(str(domains), self.name, str(dimensionSizesInDeclaration))
				)
			reorderedDomains = [0] * len(domains)
			usedIndices = []
			for (domainName, domainSize) in domains:
				index = getNextUnusedIndexForDimensionSize(domainSize, dimensionSizesInDeclaration, usedIndices)
				usedIndices.append(index)
				reorderedDomains[index] = (domainName, domainSize)
			return reorderedDomains

		if self.initLevel > Init.ROUTINENODE_ATTRIBUTES_LOADED:
			sys.stderr.write("WARNING: symbol %s's declaration is loaded when the initialization level has already advanced further.\n" \
				%(str(self))
			)

		declarationDirectives, symbolDeclarationStr = splitDeclarationSettingsFromSymbols( \
			paramDeclMatch.group(0), \
			[self], \
			patterns, \
			withAndWithoutIntent=False \
		)
		self.declarationPrefix = purgeDimensionAndGetAdjustedLine(declarationDirectives.rstrip() + " " + "::", patterns)

		#   get and check intent                                      #
		intentMatch = patterns.intentPattern.match(paramDeclMatch.group(1))
		if intentMatch and (not self.intent or self.intent.strip() == ""):
			self.intent = intentMatch.group(1)
		elif not self.intent or self.intent.strip() == "":
			self.intent = ""
		elif not self.intent == intentMatch.group(1):
			raise Exception("Symbol %s's intent was previously defined already and does not match the declaration on this line. Previously loaded intent: %s, new intent: %s" %(str(self), self.intent, intentMatch.group(1)))

		#   check whether this is a pointer
		self.isPointer = self.pointerDeclarationPattern.match(paramDeclMatch.group(0)) != None

		#   look at declaration of symbol and get its                 #
		#   dimensions.                                               #
		dimensionStr, remainder = self.getDimensionStringAndRemainderFromDeclMatch(paramDeclMatch, \
			patterns.dimensionPattern \
		)
		dimensionSizes = [sizeStr.strip() for sizeStr in dimensionStr.split(',') if sizeStr.strip() != ""]
		if self.isAutoDom and self.debugPrint:
			sys.stderr.write("reordering domains for symbol %s with autoDom option.\n" %(self.name))
		if self.isAutoDom and self.isPointer:
			if len(self.domains) == 0:
				for dimensionSize in dimensionSizes:
					self.domains.append(("HF_GENERIC_UNKNOWN_DIM", dimensionSize))
			elif len(dimensionSizes) != len(self.domains):
				raise Exception("Symbol %s's declared shape does not match its domainDependant directive. \
Automatic reshaping is not supported since this is a pointer type. Domains in Directive: %s, dimensions in declaration: %s" %(self.name, str(self.domains), str(dimensionSizes)))
		elif self.isAutoDom:
			# for the stencil use case: user will still specify the dimensions in the declaration
			# -> autodom picks them up and integrates them as parallel active dims
			for dimensionSize in dimensionSizes:
				missingParallelDomain = None
				for domName in self.aggregatedRegionDomNames:
					if self.aggregatedRegionDomSizesByName[domName] != dimensionSize:
						continue
					if domName in self.parallelActiveDims:
						continue
					missingParallelDomain = domName
					break
				if missingParallelDomain != None:
					if self.debugPrint:
						sys.stderr.write("Dimension size %s matched to a parallel region but not matched in the domain dependant \
template for symbol %s - automatically inserting it for domain name %s\n"
							%(dimensionSize, self.name, domName)
						)
					self.parallelActiveDims.append(domName)
			self.domains = []
			for parallelDomName in self.parallelActiveDims:
				parallelDomSize = self.aggregatedRegionDomSizesByName[parallelDomName]
				self.domains.append((parallelDomName, parallelDomSize))
			for dimensionSize in dimensionSizes:
				for domName in self.aggregatedRegionDomNames:
					if self.aggregatedRegionDomSizesByName[domName] == dimensionSize:
						break
				else:
					self.parallelInactiveDims.append(dimensionSize)
					self.domains.append(("HF_GENERIC_PARALLEL_INACTIVE_DIM", dimensionSize))

		# at this point we may not go further if the parallel region data
		# has not yet been analyzed.
		if self.initLevel < Init.ROUTINENODE_ATTRIBUTES_LOADED:
			if not self.isPointer:
				self.domains = getReorderedDomainsAccordingToDeclaration(self.domains, dimensionSizes)
			self.checkIntegrityOfDomains()
			return

		dimensionSizesMatchedInTemplate = []

		if not self.isPointer:
			#   compare the declared dimensions with those in the         #
			#   'parallelActive' set using the declared domain sizes.     #
			#   If there are any matches                                  #
			#   in subroutines where the parallel region is outside,      #
			#   throw an error. the user should NOT declare those         #
			#   dimensions himself.                                       #
			#   Otherwise, insert the dimensions to the declaration       #
			#   in order of their appearance in the dependant template.   #
			#   $$$ TODO: enable support for symmetric domain setups where one domain is passed in for vectorization
			lastParallelDomainIndex = -1
			for parallelDomName in self.parallelActiveDims:
				parallelDomSize = self.aggregatedRegionDomSizesByName[parallelDomName]
				if parallelDomSize in dimensionSizes and self.parallelRegionPosition == "outside":
					raise Exception("Parallel domain %s is declared for array %s in a subroutine where the parallel region is positioned outside. \
	This is not allowed. Note: These domains are inserted automatically if needed. For stencil computations it is recommended to only pass scalars to subroutine calls within the parallel region." \
						%(parallelDomName, self.name))
				if self.parallelRegionPosition == "outside":
					continue
				for index, (domName, domSize) in enumerate(self.domains):
					if domName == parallelDomName:
						lastParallelDomainIndex = index
						break
				else:
					lastParallelDomainIndex += 1
					self.domains.insert(lastParallelDomainIndex, (parallelDomName, parallelDomSize))
			if self.parallelRegionPosition == "outside":
				self.domains = [(domName, domSize) for (domName, domSize) in self.domains if not domName in self.parallelActiveDims]

		#   Now match the declared dimensions to those in the         #
		#   'parallelInactive' set, using the declared domain sizes.  #
		#   All should be matched, otherwise throw an error.          #
		#   Insert the dimensions in order of their appearance in     #
		#   the domainDependant template.                             #
		dependantDomNameAndSize = getDomNameAndSize(self.template)
		for (dependantDomName, dependantDomSize) in dependantDomNameAndSize:
			if dependantDomName not in self.parallelInactiveDims:
				continue
			if dependantDomSize not in dimensionSizes:
				raise Exception("Symbol %s's dependant non-parallel domain size %s is not declared as one of its dimensions." %(self.name, dependantDomSize))
			dimensionSizesMatchedInTemplate.append(dependantDomSize)
			if self.isPointer:
				continue
			for (domName, domSize) in self.domains:
				if dependantDomSize == domSize:
					break
			else:
				self.domains.append((dependantDomName, dependantDomSize))
		for (dependantDomName, dependantDomSize) in dependantDomNameAndSize:
			if dependantDomName not in self.parallelActiveDims:
				continue
			if dependantDomSize in dimensionSizes:
				dimensionSizesMatchedInTemplate.append(dependantDomSize)
		if self.isAutoDom and not self.isPointer:
			for dimSize in self.parallelInactiveDims:
				for (domName, domSize) in self.domains:
					if dimSize == domSize:
						break
				else:
					self.domains.append(("HF_GENERIC_PARALLEL_INACTIVE_DIM", dimSize))

		#    Sanity checks                                            #
		if len(self.domains) < len(dimensionSizes):
			raise Exception("Unexpected error: Symbol %s got less dimensions after finishing the initialization \
from the standard declaration. Declared domain: %s, Domain after init: %s, Parallel dims: %s, Independant dims: %s, \
Parallel region position: %s"
				%(self.name, str(dimensionSizes), str(self.domains), str(self.parallelActiveDims), str(self.parallelInactiveDims), self.parallelRegionPosition)
			)

		if not self.isAutoDom and len(dimensionSizes) != len(dimensionSizesMatchedInTemplate):
			raise Exception("Symbol %s's domainDependant directive does not specify the flag 'autoDom', \
but the template doesn't match all the declared dimensions. Either use the 'autoDom' attribute or specify \
all dimensions in the directive.\nNumber of declared dimensions: %i (%s); number of template dimensions: %i (%s), \
Parallel region position: %s"
				%(self.name, len(dimensionSizes), str(dimensionSizes), len(dimensionSizesMatchedInTemplate), str(dimensionSizesMatchedInTemplate), self.parallelRegionPosition)
			)
		if not self.isPointer:
			self.domains = getReorderedDomainsAccordingToDeclaration(self.domains, dimensionSizes)
		self.checkIntegrityOfDomains()
		self.initLevel = max(self.initLevel, Init.DECLARATION_LOADED)
		if self.debugPrint:
			sys.stderr.write("declaration loaded for symbol %s. Domains at this point: %s\n" %(self.name, str(self.domains)))

	def loadImportInformation(self, importMatch):
		sourceModuleName = importMatch.group(1)
		if sourceModuleName == "":
			raise Exception("Invalid module in use statement for symbol %s" %(symbol.name))
		self.sourceModule = sourceModuleName
		mapMatch = self.symbolImportMapPattern.match(importMatch.group(0))
		sourceSymbolName = ""
		if mapMatch:
			sourceSymbolName = mapMatch.group(1)
			if sourceSymbolName == "":
				raise Exception("Invalid source symbol in use statement for symbol %s" %(symbol.name))
		if sourceSymbolName == "":
			sourceSymbolName = self.name
		self.sourceSymbol = sourceSymbolName

	def getDimensionStringAndRemainderFromDeclMatch(self, paramDeclMatch, dimensionPattern):
		prefix = paramDeclMatch.group(1)
		postfix = paramDeclMatch.group(2)
		dimensionStr = ""
		remainder = ""

		dimensionMatch = dimensionPattern.match(prefix, re.IGNORECASE)
		if dimensionMatch:
			dimensionStr = dimensionMatch.group(2)
		else:
			dimensionMatch = re.match(r'\s*(?:double\s+precision\W|real\W|integer\W|logical\W).*?(?:intent\W)*.*?(?:in\W|out\W|inout\W)*.*?(?:\W|^)' + re.escape(self.name) + r'\s*\(\s*(.*?)\s*\)(.*)', \
				str(prefix + self.name + postfix), re.IGNORECASE)
			if dimensionMatch:
				dimensionStr = dimensionMatch.group(1)
				postfix = dimensionMatch.group(2)
		dimensionCheckForbiddenCharacters = re.match(r'^(?!.*[()]).*', dimensionStr, re.IGNORECASE)
		if not dimensionCheckForbiddenCharacters:
			raise Exception("Forbidden characters found in declaration of symbol %s: %s. Note: Preprocessor functions in domain dependant declarations are not allowed, only simple definitions." \
				%(self.name, dimensionStr))
		return dimensionStr, postfix

	def getAdjustedDeclarationLine(self, paramDeclMatch, parallelRegionTemplates, dimensionPattern):
		'''process everything that happens per h90 declaration symbol'''
		prefix = paramDeclMatch.group(1)
		postfix = paramDeclMatch.group(2)

		if not parallelRegionTemplates or len(parallelRegionTemplates) == 0:
			return prefix + self.deviceName() + postfix

		dimensionStr, postfix = self.getDimensionStringAndRemainderFromDeclMatch(paramDeclMatch, dimensionPattern)
		return prefix + str(self) + postfix

	def getDeclarationLineForAutomaticSymbol(self, purgeIntent=False, patterns=None):
		if self.declarationPrefix == None or self.declarationPrefix == "":
			if self.routineNode:
				routineHelperText = " for subroutine %s," %(self.routineNode.getAttribute("name"))
			raise Exception("Symbol %s needs to be automatically declared%s but there is no information about its type. \
Please specify the type like in a Fortran 90 declaration line using a @domainDependant {declarationPrefix([TYPE DECLARATION])} directive within said subroutine.\n\n\
EXAMPLE:\n\
@domainDependant {declarationPrefix(real(8))}\n\
%s\n\
@end domainDependant" %(self.automaticName(), routineHelperText, self.name)
			)

		if purgeIntent and patterns == None:
			raise Exception("Unexpected error: patterns argument required with purgeIntent argument set to True in getDeclarationLineForAutomaticSymbol.")

		declarationPrefix = self.declarationPrefix
		if "::" not in declarationPrefix:
			declarationPrefix = declarationPrefix.rstrip() + " ::"

		if purgeIntent:
			declarationDirectivesWithoutIntent, _,  symbolDeclarationStr = splitDeclarationSettingsFromSymbols(
				declarationPrefix + " " + str(self),
				[self],
				patterns,
				withAndWithoutIntent=True
			)
			declarationPrefix = declarationDirectivesWithoutIntent

		return declarationPrefix + " " + str(self)

	def automaticName(self):
		if not self.routineNode or self.declarationType() == DeclarationType.MODULE_SCALAR:
			return self.name

		referencingName = self.name + "_hfauto_" + self.routineNode.getAttribute("name")
		referencingName = referencingName.strip()
		return referencingName[:min(len(referencingName), 31)] #cut after 31 chars because of Fortran 90 limitation

	def deviceName(self):
		if self.isUsingDevicePostfix:
			return self.name + "_d"
		return self.name

	def selectAllRepresentation(self):
		if self.initLevel < Init.ROUTINENODE_ATTRIBUTES_LOADED:
			raise Exception("Symbol %s's selection representation is accessed without loading the routine node attributes first" %(str(self)))

		result = self.deviceName()
		if len(self.domains) == 0:
			return result
		result = result + "("
		for i in range(len(self.domains)):
			if i != 0:
				result = result + ","
			result = result + ":"
		result = result + ")"
		return result

	def allocationRepresentation(self):
		if self.initLevel < Init.ROUTINENODE_ATTRIBUTES_LOADED:
			raise Exception("Symbol %s's allocation representation is accessed without loading the routine node attributes first" %(str(self)))

		result = self.deviceName()
		if len(self.domains) == 0:
			return result
		result += "("
		domPP = self.domPP()
		if self.numOfParallelDomains != 0 and domPP != "":
			result += domPP + "("
		for index, domain in enumerate(self.domains):
			if index != 0:
				result = result + ","
			dimSize = domain[1]
			if dimSize == ":":
				raise Exception("Cannot generate allocation call for symbol %s on the device - one or more dimension sizes are unknown at this point. \
Please specify the domains and their sizes with domName and domSize attributes in the corresponding @domainDependant directive." %(self.name))
			result += dimSize
		if self.numOfParallelDomains != 0 and domPP != "":
			result += ")"
		result += ")"
		return result

	def accessRepresentation(self, parallelIterators, offsets):
		if self.debugPrint:
			sys.stderr.write("producing access representation for symbol %s; parallel iterators: %s, offsets: %s\n" %(self.name, str(parallelIterators), str(offsets)))

		if self.initLevel < Init.ROUTINENODE_ATTRIBUTES_LOADED:
			return self.name

		if len(parallelIterators) == 0 and len(offsets) != len(self.domains) - self.numOfParallelDomains \
		and len(offsets) != len(self.domains):
			raise Exception("Unexpected number of offsets specified for symbol %s; Offsets: %s, Expected domains: %s" \
				%(self.name, offsets, self.domains))
		elif len(parallelIterators) != 0 and len(offsets) + len(parallelIterators) != len(self.domains) \
		and len(offsets) != len(self.domains):
			raise Exception("Unexpected number of offsets and iterators specified for symbol %s; Offsets: %s, Iterators: %s, Expected domains: %s" \
				%(self.name, offsets, parallelIterators, self.domains))

		result = self.deviceName()
		if self.isAutomatic:
			result = self.automaticName()

		if len(self.domains) == 0:
			return result
		result = result + "("
		accPP = self.accPP()
		if self.numOfParallelDomains != 0 and accPP != "":
			result = result + accPP + "("
		nextOffsetIndex = 0
		for i in range(len(self.domains)):
			if i != 0:
				result = result + ","
			if len(parallelIterators) == 0 and len(offsets) == len(self.domains):
				result = result + offsets[i]
				continue
			elif len(parallelIterators) == 0 \
			and len(offsets) == len(self.domains) - self.numOfParallelDomains \
			and i < self.numOfParallelDomains:
				result = result + ":"
				continue
			elif len(parallelIterators) == 0 \
			and len(offsets) == len(self.domains) - self.numOfParallelDomains \
			and i >= self.numOfParallelDomains:
				result = result + offsets[i - self.numOfParallelDomains]
				continue

			#if we reach this there are parallel iterators specified.
			if self.domains[i][0] in parallelIterators:
				result += self.domains[i][0]
			elif nextOffsetIndex < len(offsets):
				result += offsets[nextOffsetIndex]
				nextOffsetIndex += 1
			else:
				raise Exception("Cannot generate access representation for symbol %s: Unknown parallel iterators specified (%s) or not enough offsets (%s)."
					%(str(self), str(parallelIterators), str(offsets))
				)

		if self.numOfParallelDomains != 0 and accPP != "":
			result = result + ")"
		result = result + ")"
		return result

	def declarationType(self):
		if self.sourceModule == "HF90_LOCAL_MODULE":
			return DeclarationType.MODULE_SCALAR
		if self.sourceModule != None and self.sourceModule != "":
			return DeclarationType.IMPORTED_SCALAR
		if self.initLevel < Init.ROUTINENODE_ATTRIBUTES_LOADED:
			return DeclarationType.UNDEFINED
		if self.intent == "" and len(self.domains) > 0:
			return DeclarationType.LOCAL_ARRAY
		return DeclarationType.OTHER


	def getTemplateEntryNodeValues(self, parentName):
		if not self.template:
			return None
		parentNodes = self.template.getElementsByTagName(parentName)
		if not parentNodes or len(parentNodes) == 0:
			return None
		return [entry.firstChild.nodeValue for entry in parentNodes[0].childNodes]

	def getDeclarationMatch(self, line):
		match = self.declPattern.match(line)
		if not match:
			return None
		#check whether the symbol is matched inside parenthesis - it could be part of the dimension definition
		#if it is indeed part of a dimension we can forget it and return None - according to Fortran definition
		#cannot be declared as its own dimension.
		analyzer = BracketAnalyzer()
		if analyzer.currLevelAfterString(match.group(1)) != 0:
			return None
		else:
			return match

	def domPP(self):
		domPPEntries = self.getTemplateEntryNodeValues("domPP")
		if domPPEntries and len(domPPEntries) > 0:
			return domPPEntries[0]

		if self.isAutoDom:
			numOfDimensions = len(self.domains)
			domPPName = ""
			if numOfDimensions < 3:
				domPPName = ""
			elif numOfDimensions == 3:
				domPPName = "DOM"
			else:
				domPPName = "DOM%i" %(numOfDimensions)
			return domPPName
		else:
			return ""


	def accPP(self):
		accPPEntries = self.getTemplateEntryNodeValues("accPP")
		if accPPEntries and len(accPPEntries) > 0:
			return accPPEntries[0]

		if self.isAutoDom:
			numOfDimensions = len(self.domains)
			accPPName = ""
			if numOfDimensions < 3:
				accPPName = ""
			elif numOfDimensions == 3:
				accPPName = "AT"
			else:
				accPPName = "AT%i" %(numOfDimensions)
			return accPPName
		else:
			return ""

class FrameworkArray(Symbol):

	def __init__(self, name, declarationPrefix, domains, isOnDevice):
		if not name or name == "":
			raise Exception("Unexpected error: name required for initializing framework array")
		if not declarationPrefix or declarationPrefix == "":
			raise Exception("Unexpected error: declaration prefix required for initializing framework array")
		if len(domains) != 1:
			raise Exception("Unexpected error: currently unsupported non-1D-array specified as framework array")

		self.name = name
		self.domains = domains
		self.isMatched = True
		self.isAutomatic = True
		self.isOnDevice = isOnDevice
		self.declarationPrefix = declarationPrefix
		self.initLevel = Init.NOTHING_LOADED

	def declarationType(self):
		return DeclarationType.FRAMEWORK_ARRAY


