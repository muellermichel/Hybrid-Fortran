#!/usr/bin/python
# -*- coding: UTF-8 -*-

# Copyright (C) 2014 Michel Müller, Tokyo Institute of Technology

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
#  Procedure        H90Symbol.py                                       #
#  Comment          Provide functionality for HF symbols               #
#  Date             2012/08/02                                         #
#  Author           Michel Müller (AOKI Laboratory)                    #
#**********************************************************************#

import re, sys, copy
import logging
import pdb
from DomHelper import *
from GeneralHelper import enum, BracketAnalyzer
from H90RegExPatterns import H90RegExPatterns
from GeneralHelper import Singleton
from H90SymbolDependencyGraphAnalyzer import SymbolDependencyAnalyzer, SymbolType

Init = enum("NOTHING_LOADED",
	"TEMPLATE_LOADED",
	"DEPENDANT_ENTRYNODE_ATTRIBUTES_LOADED",
	"ROUTINENODE_ATTRIBUTES_LOADED",
	"DECLARATION_LOADED"
)

#   Boxes = Symbol States
#   X -> Transition Texts = Methods being called by parser
#   Other Texts = Helper Functions
#
#                                        +----------------+
#                                        | NOTHING_LOADED |
#                                        +------+---------+
#                                               |
#                                         loadTemplate
#                                               |
#                                        +------v----------+
#                                        | TEMPLATE_LOADED |
#                                        +------+----------+
#                                               |                 X -> (routine entry)
#                                               |  loadDomainDependantEntryNodeAttributes
#                                               |                              ^
#                              +----------------v-----------------------+      |
#                              | DEPENDANT_ENTRYNODE_ATTRIBUTES_LOADED  |      | (module entry)
#                              +----+-----------+-----------------------+      |
#                     X ->          +           |           X ->               |
# +-------------+  loadRoutineNodeAttributes    |   loadModuleNodeAttributes  +-----------+
# |                                 +           |                              |          |
# |                                 |           |                              |          |
# |                                +v-----------v------------------+           |          |
# |                                | ROUTINENODE_ATTRIBUTES_LOADED |    +------+          |
# |                                +---+--------+------------------+    |                 |
# |                                    |        |            X ->       +                 |
# |                                    |        |    loadImportInformation+----------------------------------+
# |             X -> loadDeclaration   |        |               +      +                  |                  |
# |                       +            |        |               |      |                  |                  |
# |                       |           +v--------v-----------+   |      |                  |                  |
# |                       |           | DECLARATION_LOADED  |   |      |                  |                  |
# |                       |           +---------------------+   |      |                  |                  |
# |                       |                                     |      |                  |                  |
# |                       |                                     |      |                  |                  |
# |                       v                                     v      |                  |                  |
# |                       getReorderedDomainsAccordingToDeclaration    |                  |                  |
# |                                                                    |                  |                  |
# |                                                                    |                  |                  |
# |                                                                    |                  |                  |
# |                                                                    |                  |                  |
# |                                                                    |                  |                  |
# +-----------------------------------> loadTemplateAttributes <--------------------------+                  |
#                                         +       +                    |                                     |
#                                         |       |                    |                                     |
#                                         |       |                    |                                     |
#                                         |       |                    |                                     |
#                                         |       |                    |                                     v
#   loadDomains <-------------------------+       +----------------------------------------------------->  loadDeclarationPrefixFromString
#        ^                                                             |
#        |                                                             |
#        |                                                             |
#        |                                                             |
#        +-------------------------------------------------------------+



#Set of declaration types that are mutually exlusive for
#declaration lines in Hybrid Fortran.
#-> You cannot mix and match declarations of different types
DeclarationType = enum("UNDEFINED",
	"LOCAL_ARRAY",
	"LOCAL_MODULE_SCALAR",
	"MODULE_ARRAY",
	"FOREIGN_MODULE_SCALAR",
	"FRAMEWORK_ARRAY",
	"MODULE_ARRAY_PASSED_IN_AS_ARGUMENT",
	"OTHER_ARGUMENT",
	"OTHER_ARRAY",
	"OTHER_SCALAR",
	"LOCAL_SCALAR"
)

def purgeFromDeclarationSettings(line, dependantSymbols, patterns, purgeList=['intent'], withAndWithoutIntent=True):
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

	purgedDeclarationDirectives = declarationDirectives

	for keywordToPurge in purgeList:
		match = re.match(r"(.*?)\s*(,?)\s*" + keywordToPurge + r"\s*\(.*?\)\s*(,?)\s*(.*)", purgedDeclarationDirectives, re.IGNORECASE)
		if not match:
			match = re.match(r"(.*?)\s*(,?)\s*" + keywordToPurge + r"\s*(,?)\s*(.*)", purgedDeclarationDirectives, re.IGNORECASE)
		if match:
			sepChar = ", " if match.group(2) != "" and match.group(3) != "" else " "
			purgedDeclarationDirectives = match.group(1) + sepChar + match.group(4)

	return purgedDeclarationDirectives, declarationDirectives, symbolDeclarationStr

def getReorderedDomainsAccordingToDeclaration(domains, dimensionSizesInDeclaration, purgeUndeclared=False):
	def getNextUnusedIndexForDimensionSize(domainSize, dimensionSizesInDeclaration, usedIndices):
		index_candidate = None
		startAt = 0
		while True:
			if startAt > len(dimensionSizesInDeclaration) - 1:
				return None
			try:
				index_candidate = dimensionSizesInDeclaration[startAt:].index(domainSize) + startAt
			except ValueError:
				return None #example: happens when domains are declared for allocatables with :
			if index_candidate in usedIndices:
				startAt = index_candidate + 1
			else:
				break;
		return index_candidate

	if not dimensionSizesInDeclaration:
		return domains
	if len(domains) == 0:
		return domains
	if len(domains) != len(dimensionSizesInDeclaration) and not purgeUndeclared:
		return domains
	# if purgeUndeclared:
	#     newDomains = []
	#     for (domainName, domainSize) in domains:
	#         if domainSize in dimensionSizesInDeclaration:
	#             newDomains.append((domainName, domainSize))
	#     domains = newDomains
	reorderedDomains = [0] * len(domains)
	usedIndices = []
	fallBackToCurrentOrder = False
	for (domainName, domainSize) in domains:
		index = getNextUnusedIndexForDimensionSize(domainSize, dimensionSizesInDeclaration, usedIndices)
		if index == None:
			fallBackToCurrentOrder = True
			break
		usedIndices.append(index)
		reorderedDomains[index] = (domainName, domainSize)
	if fallBackToCurrentOrder:
		return domains
	return reorderedDomains

def purgeDimensionAndGetAdjustedLine(line, patterns):
	match = patterns.dimensionPattern.match(line)
	if not match:
		return line
	else:
		return match.group(1) + match.group(3)

def uniqueIdentifier(symbolName, nameOfScope):
	return (symbolName + "_hfauto_" + nameOfScope).strip()

MERGEABLE_DEFAULT_SYMBOL_INSTANCE_ATTRIBUTES = {
	"isPointer": False,
	"isMatched": False,
	"declarationPrefix": None,
	"sourceModule": None,
	"sourceSymbol": None,
	"parallelRegionTemplates": None,
	"declaredDimensionSizes": None,
	"_isPresent": False,
	"isAutoDom": False, #MMU 2015-11-6: At this point we can still not make autoDom the default. It would generate the following error in ASUCA:
#                        Error when parsing file /work1/t2g-kaken-S/mueller/asuca/hybrid/asuca-kij/./build/hf_preprocessed/physics.h90 on line 475:
#                        There are multiple known dimension sizes for domain i. Cannot insert domain for autoDom symbol densrjd. Please use explicit declaration;
#                        Debug Print: None; Print of line:
#                        real(rp):: densrjd(nz)
	"_isToBeTransfered": False,
	"_isHostSymbol": False,
	"isCompacted": False,
	"domPPName": None,
	"accPPName": None,
	"analysis": None,
	"_declarationTypeOverride": None
}

MERGEABLE_DEFAULT_SYMBOL_INSTANCE_DOMAIN_ATTRIBUTES = {
	"domains": [],
	"parallelActiveDims": [], #!Important: The order of this list must remain insignificant when it is used
	"parallelInactiveDims": [], #!Important: The order of this list must remain insignificant when it is used
	"aggregatedRegionDomSizesByName": {},
	"aggregatedRegionDomNames": []
}

class Symbol(object):
	def __init__(self, name, template=None, patterns=None, symbolEntry=None, scopeNode=None, analysis=None, parallelRegionTemplates=[]):
		if not name or name == "":
			raise Exception("Name required for initializing symbol")

		self.loadDefaults()
		self.name = name
		if patterns != None:
			self.patterns = patterns
		else:
			self.patterns = H90RegExPatterns.Instance()
		self.analysis = analysis
		self.declPattern = self.patterns.get(r'(\s*(?:double\s+precision|real|integer|logical).*?[\s,:]+)' + re.escape(name) + r'((?:\s|\,|\(|$)+.*)')
		self.symbolImportPattern = self.patterns.get(r'^\s*use\s*(\w*)[,\s]*only\s*\:.*?\W' + re.escape(name) + r'\W.*')
		self.symbolImportMapPattern = self.patterns.get(r'.*?\W' + re.escape(name) + r'\s*\=\>\s*(\w*).*')
		self.pointerDeclarationPattern = self.patterns.get(r'\s*(?:double\s+precision|real|integer|logical).*?pointer.*?[\s,:]+' + re.escape(name))
		self.initLevel = Init.NOTHING_LOADED
		self.routineNode = None
		if template != None:
			self.loadTemplate(template)
		else:
			self.template = None
		if (symbolEntry != None or scopeNode != None) \
		and (symbolEntry == None or scopeNode == None):
			raise Exception("symbol entry (%s) and scope node (%s) need to be set together for symbol %s" %(symbolEntry, scopeNode, self))
		if symbolEntry != None:
			self.isModuleSymbol = scopeNode.tagName == "module"
			self.loadDomainDependantEntryNodeAttributes(symbolEntry)
			if self.isModuleSymbol:
				self.loadModuleNodeAttributes(scopeNode)
			else:
				self.loadRoutineNodeAttributes(scopeNode, parallelRegionTemplates)
		logging.debug("[" + self.name + ".init " + str(self.initLevel) + "] initialized")

	def __repr__(self):
		return self.domainRepresentation()

	def __eq__(self, other):
		if other == None:
			return False
		return self.nameInScope() == other.nameInScope()
	def __ne__(self, other):
		if other == None:
			return True
		return self.nameInScope() != other.nameInScope()
	def __lt__(self, other):
		if other == None:
			return False
		return self.nameInScope() < other.nameInScope()
	def __le__(self, other):
		if other == None:
			return False
		return self.nameInScope() <= other.nameInScope()
	def __gt__(self, other):
		if other == None:
			return True
		return self.nameInScope() > other.nameInScope()
	def __ge__(self, other):
		if other == None:
			return False
		return self.nameInScope() >= other.nameInScope()

	@property
	def nameIsGuaranteedUniqueInScope(self):
		#symbols without a routineNode are module scope symbols
		if not self.routineNode or self.isArgument or self.declarationType in [
			DeclarationType.LOCAL_MODULE_SCALAR
		]:
			return True
		return False

	@property
	def isArgument(self):
		return self._isArgumentOverride or (self.analysis and self.analysis.symbolType in [SymbolType.ARGUMENT_WITH_DOMAIN_DEPENDANT_SPEC, SymbolType.ARGUMENT])

	@property
	def namePattern(self):
		symbolName = self._nameInScope if self._nameInScope != None else self.name
		return self.patterns.get(r'((?:[^\"\']|(?:\".*\")|(?:\'.*\'))*?(?:\W|^))(' + re.escape(symbolName) + r'(?:_d)?)((?:\W.*)|\Z)')

	@isArgument.setter
	def isArgument(self, _isArgumentOverride):
		self._isArgumentOverride = _isArgumentOverride

	@property
	def isUsingDevicePostfix(self):
		return self._isUsingDevicePostfix

	@isUsingDevicePostfix.setter
	def isUsingDevicePostfix(self, _isUsingDevicePostfix):
		self._isUsingDevicePostfix = _isUsingDevicePostfix

	@property
	def nameOfScope(self):
		if self._nameOfScopeOverride:
			return self._nameOfScopeOverride
		if self.routineNode:
			return self.routineNode.getAttribute('name')
		return None

	@nameOfScope.setter
	def nameOfScope(self, _nameOfScopeOverride):
		self._nameOfScopeOverride = _nameOfScopeOverride

	@property
	def isHostSymbol(self):
		return self._isHostSymbol and not self._isPresent and not self._isToBeTransfered

	@isHostSymbol.setter
	def isHostSymbol(self, _isHostSymbol):
		self._isHostSymbol = _isHostSymbol

	@property
	def isPresent(self):
		return self._isPresent

	@property
	def isArray(self):
		if self.domains and len(self.domains) > 0:
			return True
		return self.declarationType in [
			DeclarationType.LOCAL_ARRAY,
			DeclarationType.MODULE_ARRAY,
			DeclarationType.FRAMEWORK_ARRAY,
			DeclarationType.MODULE_ARRAY_PASSED_IN_AS_ARGUMENT,
			DeclarationType.OTHER_ARRAY
		]

	@property
	def isToBeTransfered(self):
		return self._isToBeTransfered

	@property
	def numOfParallelDomains(self):
		if self.parallelRegionPosition == "outside":
			return 0
		return len(self.parallelActiveDims)

	@property
	def activeDomainsMatchSpecification(self):
		if not self.domains:
			return False
		if self.template:
			templateDomains = getDomNameAndSize(self.template)
			#check whether domains are specified in the domainDependant directive
			#otherwise autoDom is assumed for everything and we just have to compare against the Fortran declaration
			if len(templateDomains) > 0:
				if len(self.domains) == len(templateDomains):
					return True #all domains are explicitely declared in the domainDependant directive
				if len(self.domains) == len(templateDomains) + len(self.parallelInactiveDims):
					return True #parallel domains are explicitely declared in the domainDependant directive
				return False #domainDependant directive specification is not active here (probably the parallelRegion(s) they are meant for do not apply currently)
		if self.declaredDimensionSizes and len(self.domains) == len(self.declaredDimensionSizes):
			return True
		return False

	@property
	def declarationType(self):
		def hasSourceModule(symbol, onlyLocal=False):
			if symbol.sourceModule == "HF90_LOCAL_MODULE" \
			or (symbol.sourceModule not in [None, ""] and symbol.sourceModule == self.routineNode.getAttribute('module')):
				return True
			if onlyLocal:
				return False
			if symbol.sourceModule not in [None, ""]:
				return True
			return False

		if self._declarationTypeOverride != None:
			return self._declarationTypeOverride
		if not self.routineNode:
			raise Exception("Cannot define declaration type for symbol %s without a routine or module node loaded" %(self))
		if len(self.domains) > 0:
			if hasSourceModule(self):
				if self.isArgument:
					return DeclarationType.MODULE_ARRAY_PASSED_IN_AS_ARGUMENT
				return DeclarationType.MODULE_ARRAY
			if self.isArgument:
				return DeclarationType.OTHER_ARGUMENT
			if self.initLevel < Init.ROUTINENODE_ATTRIBUTES_LOADED:
				return DeclarationType.UNDEFINED
			if self.intent == "local":
				return DeclarationType.LOCAL_ARRAY
			return DeclarationType.OTHER_ARRAY
		if self.isArgument:
			return DeclarationType.OTHER_ARGUMENT
		if hasSourceModule(self, onlyLocal=True):
			return DeclarationType.LOCAL_MODULE_SCALAR
		if hasSourceModule(self):
			return DeclarationType.FOREIGN_MODULE_SCALAR
		if self.intent == "local":
			return DeclarationType.LOCAL_SCALAR
		if self.initLevel < Init.ROUTINENODE_ATTRIBUTES_LOADED:
			return DeclarationType.UNDEFINED
		return DeclarationType.OTHER_SCALAR

	@declarationType.setter
	def declarationType(self, _declarationTypeOverride):
		self._declarationTypeOverride = _declarationTypeOverride

	def _getPurgedDeclarationPrefix(self, purgeList=[]):
		if self.declarationPrefix == None or self.declarationPrefix == "":
			if self.routineNode:
				routineHelperText = " for subroutine %s," %(self.nameOfScope)
			raise Exception("Symbol %s (type %i) needs to be automatically declared%s but there is no information about its type. \
Please either use an @domainDependant specification in the imported module's module scope OR \
specify the type like in a Fortran 90 declaration line using a @domainDependant {declarationPrefix([TYPE DECLARATION])} directive within the current subroutine.\n\n\
EXAMPLE:\n\
@domainDependant {declarationPrefix(real(8))}\n\
%s\n\
@end domainDependant" %(self.nameInScope(), self.declarationType, routineHelperText, self.name)
			)

		declarationPrefix = self.declarationPrefix
		if "::" not in declarationPrefix:
			declarationPrefix = declarationPrefix.rstrip() + " ::"
		if len(purgeList) != 0:
			patterns = H90RegExPatterns.Instance()
			declarationDirectivesWithoutIntent, _,  symbolDeclarationStr = purgeFromDeclarationSettings(
				declarationPrefix + " " + str(self),
				[self],
				patterns,
				purgeList=purgeList,
				withAndWithoutIntent=True
			)
			declarationPrefix = declarationDirectivesWithoutIntent
		return declarationPrefix.strip()

	@property
	def uniqueIdentifier(self):
		return uniqueIdentifier(self.name, self.nameOfScope)

	def nameInScope(self, useDeviceVersionIfAvailable=True):
		#Give a symbol representation that is guaranteed to *not* collide with any local namespace (as long as programmer doesn't use any 'hfXXX' pre- or postfixes)
		def automaticName(symbol):
			if symbol.analysis and symbol.routineNode:
				aliasName = symbol.analysis.aliasNamesByRoutineName.get(symbol.nameOfScope)
				if aliasName not in [None, '']:
					return aliasName
			if symbol.nameIsGuaranteedUniqueInScope:
				return symbol.name
			referencingName = symbol.uniqueIdentifier
			return referencingName[:min(len(referencingName), 31)] #cut after 31 chars because of Fortran 90 limitation

		if self.isEmulatingSymbolThatWasActiveInCurrentScope:
			self._nameInScope = self.name
		if self._nameInScope == None:
			self._nameInScope = automaticName(self)
		if useDeviceVersionIfAvailable and self.isUsingDevicePostfix:
			return self._nameInScope + "_d"
		return self._nameInScope

	def isDummySymbolForRoutine(self, routineName):
		if not self.analysis:
			return False
		return self.analysis.argumentIndexByRoutineName.get(routineName) != None

	def loadDefaults(self):
		def loadAttributesFromObject(obj):
			for attribute in obj:
				setattr(self, attribute, obj[attribute])

		self.intent = None
		self.isConstant = False
		self.routineNode = None
		self.attributes = None
		self.parallelRegionPosition = None
		self._isUsingDevicePostfix = False
		self.isOnDevice = False
		self.isModuleSymbol = False
		self._isArgumentOverride = False
		self._nameOfScopeOverride = None
		self._nameInScope = None
		self.isEmulatingSymbolThatWasActiveInCurrentScope = False
		loadAttributesFromObject(MERGEABLE_DEFAULT_SYMBOL_INSTANCE_ATTRIBUTES)
		loadAttributesFromObject(MERGEABLE_DEFAULT_SYMBOL_INSTANCE_DOMAIN_ATTRIBUTES)

	def merge(self, otherSymbol):
		def getMergedSimpleAttributeValue(attributeName):
			mine = getattr(self, attributeName)
			other = getattr(otherSymbol, attributeName)
			if mine != other \
			and mine not in ["", MERGEABLE_DEFAULT_SYMBOL_INSTANCE_ATTRIBUTES[attributeName]] \
			and other not in ["", MERGEABLE_DEFAULT_SYMBOL_INSTANCE_ATTRIBUTES[attributeName]]:
				return mine
			if other not in ["", MERGEABLE_DEFAULT_SYMBOL_INSTANCE_ATTRIBUTES[attributeName]]:
				return other
			return mine

		def getMergedDomains():
			mine = self.domains
			other = otherSymbol.domains
			if len(mine) > 0 and len(other) > 0 and len(mine) != len(other):
				return mine
			if len(mine) > 0 and len(other) > 0:
				for index, entry in enumerate(mine):
					if entry != other[index]:
						break
					else:
						return mine
				merged = []
				for index, entry in enumerate(mine):
					if entry[1] != other[index][1]:
						break
					myDomName = entry[0]
					otherDomName = other[index][0]
					if myDomName == otherDomName:
						merged.append(entry)
					elif myDomName == 'HF_GENERIC_DIM':
						merged.append(other[index])
					elif otherDomName == 'HF_GENERIC_DIM':
						merged.append(entry)
					else:
						break
				else:
					return merged
				return mine
			if len(other) > 0:
				return other
			return mine

		def getMergedCollection(attributeName):
			mine = getattr(self, attributeName)
			other = getattr(otherSymbol, attributeName)
			if type(mine) != type(other):
				raise Exception("Symbol %s in (%s|%s): Type conflict with attribute %s: %s vs %s." %(
					self.name,
					self.nameOfScope,
					otherSymbol.nameOfScope,
					attributeName,
					type(mine),
					type(other)
				))
			if isinstance(mine, dict):
				if len(mine.keys()) > 0 and len(other.keys()) > 0:
					return mine
				if len(other.keys()) > 0:
					return other
				return mine
			if not isinstance(mine, list):
				raise Exception("unrecognized collection type for attribute %s" %(attributeName))
			if len(mine) > 0 and len(other) > 0:
				return mine
			if len(other) > 0:
				return other
			return mine

		def isEmpty(obj):
			if isinstance(obj, dict):
				return len(obj.keys()) == 0
			if obj == None:
				return True
			return len(obj) == 0

		def allAttributesEmpty(obj, attributeList):
			for attribute in attributeList:
				entry = getattr(obj, attribute)
				if not isEmpty(entry):
					break
			else:
				return True
			return False

		if self.name != otherSymbol.name and self.sourceSymbol != otherSymbol.name and self.name != otherSymbol.sourceSymbol:
			raise Exception("cannot merge %s with %s - doesn't seem to be be the same symbol" %(self.name, otherSymbol.name))
		for attribute in MERGEABLE_DEFAULT_SYMBOL_INSTANCE_ATTRIBUTES:
			setattr(self, attribute, getMergedSimpleAttributeValue(attribute))
		for domainAttributeName in MERGEABLE_DEFAULT_SYMBOL_INSTANCE_DOMAIN_ATTRIBUTES.keys():
			if domainAttributeName == "domains":
				self.domains = getMergedDomains()
			else:
				setattr(self, domainAttributeName, getMergedCollection(domainAttributeName))
		self.initLevel = max(self.initLevel, otherSymbol.initLevel)

	def loadTemplate(self, template):
		if self.initLevel > Init.DEPENDANT_ENTRYNODE_ATTRIBUTES_LOADED:
			raise Exception(
				"Cannot load a new template for symbol %s at init level %s" %(
					str(self),
					self.initLevel
				)
			)
		self.template = template
		self.attributes = getAttributes(self.template)
		self.setOptionsFromAttributes(self.attributes)
		self.initLevel = max(self.initLevel, Init.TEMPLATE_LOADED)

	# to be called when a previously loaded symbol is used in a new scope
	def resetScope(self, newScopeName):
		self._nameOfScopeOverride = newScopeName
		self._nameInScope = None

	def setOptionsFromAttributes(self, attributes):
		if "present" in attributes:
			self._isPresent = True
		if "autoDom" in attributes:
			self.isAutoDom = True
		if "host" in attributes:
			self._isHostSymbol = True
		if "transferHere" in attributes:
			if self._isPresent:
				raise Exception("Symbol %s has contradicting attributes 'transferHere' and 'present'" %(self))
			self._isToBeTransfered = True
		logging.debug("[" + self.name + ".init " + str(self.initLevel) + "] attributes set")

	def storeDomainDependantEntryNodeAttributes(self, domainDependantEntryNode):
		logging.debug("[" + self.name + ".init " + str(self.initLevel) + "] storing symbol attributes. Init Level: %s" %(str(self.initLevel)))
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

	def loadDomainDependantEntryNodeAttributes(self, domainDependantEntryNode, warnOnOverwrite=True):
		logging.debug("[" + self.name + ".init " + str(self.initLevel) + "] +++++++++ LOADING DOMAIN DEPENDANT NODE ++++++++++ ")

		#   This symbol has an explicit domain dependant entry - make sure to store this as the name used in the scope
		self._nameInScope = self.name

		self.intent = domainDependantEntryNode.getAttribute("intent") if self.intent in [None, ''] else self.intent
		self.declarationPrefix = domainDependantEntryNode.getAttribute("declarationPrefix") if self.declarationPrefix in [None, ''] else self.declarationPrefix
		self.sourceModule = domainDependantEntryNode.getAttribute("sourceModule") if self.sourceModule in [None, ''] else self.sourceModule
		self.sourceSymbol = domainDependantEntryNode.getAttribute("sourceSymbol") if self.sourceSymbol in [None, ''] else self.sourceSymbol
		if self.isModuleSymbol:
			self.sourceModule = "HF90_LOCAL_MODULE" if self.sourceModule in [None, ''] else self.sourceModule
		self.isPointer = domainDependantEntryNode.getAttribute("isPointer") == "yes" if not self.isPointer else self.isPointer
		declaredDimensionSizes = domainDependantEntryNode.getAttribute("declaredDimensionSizes")
		self.declaredDimensionSizes = declaredDimensionSizes.split(",") if declaredDimensionSizes \
			and declaredDimensionSizes != "" \
			and self.declaredDimensionSizes == None \
			else self.declaredDimensionSizes
		if self.declaredDimensionSizes and len(self.declaredDimensionSizes) > 0:
			self.domains = []
			for dimSize in self.declaredDimensionSizes:
				if dimSize.strip() != "":
					self.domains.append(('HF_GENERIC_DIM', dimSize))
			logging.debug("[" + self.name + ".init " + str(self.initLevel) + "] dimsizes from domain dependant node: %s " %(str(self.declaredDimensionSizes)))
		self.checkIntegrityOfDomains()
		self.initLevel = max(self.initLevel, Init.DEPENDANT_ENTRYNODE_ATTRIBUTES_LOADED)

	def checkIntegrityOfDomains(self):
		if self.declaredDimensionSizes:
			for dimensionSize in self.declaredDimensionSizes:
				if not type(dimensionSize) in [str, unicode] or dimensionSize == "":
					raise Exception("Invalid definition of dimension size in symbol %s: %s" %(self.name, dimensionSize))
		for domain in self.domains:
			if not isinstance(domain, tuple):
				raise Exception("Invalid definition of domain in symbol %s: %s" %(self.name, str(domain)))
		parallelDomainSizesDict = {}
		for domainName, domainSize in self.domains:
			if not domainName in self.parallelActiveDims:
				continue
			if domainSize in parallelDomainSizesDict:
				raise Exception("Multiple occurences of parallel dimension %s for symbol %s - this is not allowed in Hybrid Fortran" %(
					domainSize,
					str(self)
				))
			parallelDomainSizesDict[domainSize] = None

	def loadTemplateAttributes(self, parallelRegionTemplates=[]):
		if self.initLevel < Init.TEMPLATE_LOADED:
			raise Exception(
				"Cannot load template attributes for %s at init level %s" %(
					str(self),
					self.initLevel
				)
			)
		dependantDomNameAndSize = getDomNameAndSize(self.template)
		declarationPrefixFromTemplate = getDeclarationPrefix(self.template)
		self.loadDeclarationPrefixFromString(declarationPrefixFromTemplate)
		self.loadDomains(dependantDomNameAndSize, parallelRegionTemplates)
		logging.debug(
				"[" + str(self) + ".init " + str(self.initLevel) + "] Domains loaded from callgraph information for symbol %s. Parallel active: %s. Parallel Inactive: %s. Declaration Prefix: %s. dependantDomNameAndSize: %s declarationPrefix: %s. Parallel Regions: %i\n" %(
					str(self),
					str(self.parallelActiveDims),
					str(self.parallelInactiveDims),
					declarationPrefixFromTemplate,
					dependantDomNameAndSize,
					declarationPrefixFromTemplate,
					len(parallelRegionTemplates)
				)
			)

	def loadDeclarationPrefixFromString(self, declarationPrefixFromTemplate):
		if declarationPrefixFromTemplate != None and declarationPrefixFromTemplate.strip() != "":
			self.declarationPrefix = declarationPrefixFromTemplate
		logging.debug("[" + self.name + ".init " + str(self.initLevel) + "] declaration prefix loaded: %s" %(declarationPrefixFromTemplate))

	def loadDomains(self, dependantDomNameAndSize, parallelRegionTemplates=[]):
		if dependantDomNameAndSize == None or len(dependantDomNameAndSize) == 0:
			dependantDomSizeByName = dict(
				("%s_%i" %(value[0],index),value[1])
				for index,value
				in enumerate(self.domains)
			) #in case we have generic domain names, need to include the index here so the order doesn't get messed up.
		else:
			dependantDomSizeByName = dict(
				(dependantDomName,dependantDomSize)
				for (dependantDomName, dependantDomSize)
				in dependantDomNameAndSize
			)
		#   which of those dimensions are invariants in               #
		#   the currently active parallel regions?                    #
		#   -> put them in the 'parallelActive' set, put the          #
		#   others in the 'parallelInactive' set.                     #
		self.parallelActiveDims = []
		self.parallelInactiveDims = []
		self.aggregatedRegionDomNames = []
		self.aggregatedRegionDomSizesByName = {}
		parallelRegionDomNamesBySize = {}
		for parallelRegionTemplate in parallelRegionTemplates:
			regionDomNameAndSize = getDomNameAndSize(parallelRegionTemplate)
			logging.debug("[" + self.name + ".init " + str(self.initLevel) + "] analyzing domains for parallel region: %s; dependant domsize by name: %s" %(
				str(regionDomNameAndSize),
				str(dependantDomSizeByName)
			))
			for (regionDomName, regionDomSize) in regionDomNameAndSize:
				if regionDomName in dependantDomSizeByName.keys() and regionDomName not in self.parallelActiveDims:
					self.parallelActiveDims.append(regionDomName)
				#The same domain name can sometimes have different domain sizes used in different parallel regions, so we build up a list of these sizes.
				if not regionDomName in self.aggregatedRegionDomSizesByName:
					self.aggregatedRegionDomSizesByName[regionDomName] = [regionDomSize]
				elif regionDomSize not in self.aggregatedRegionDomSizesByName[regionDomName]:
					self.aggregatedRegionDomSizesByName[regionDomName].append(regionDomSize)
				self.aggregatedRegionDomNames.append(regionDomName)
				parallelRegionDomNamesBySize[regionDomSize] = regionDomName

		for (dependantDomName, dependantDomSize) in dependantDomNameAndSize:
			#build up parallel inactive dimensions again
			if dependantDomName not in self.parallelActiveDims:
				self.parallelInactiveDims.append(dependantDomName)
			#use the declared domain size (potentially overriding automatic sizes)
			if dependantDomName in self.aggregatedRegionDomSizesByName:
				self.aggregatedRegionDomSizesByName[dependantDomName][0] = dependantDomSize

		logging.debug("[" + self.name + ".init " + str(self.initLevel) + "] before reset. parallel active: %s; parallel inactive: %s" %(
				str(self.parallelActiveDims),
				str(self.parallelInactiveDims)
			))

		dimsBeforeReset = copy.deepcopy(self.domains)
		self.domains = []
		for (dependantDomName, dependantDomSize) in dependantDomNameAndSize:
			if dependantDomName not in self.parallelActiveDims \
			and dependantDomName not in self.parallelInactiveDims:
				raise Exception("Symbol %s's dependant domain size %s is not declared as one of its dimensions." \
					%(self.name, dependantDomSize))
			self.domains.append((dependantDomName, dependantDomSize))
			logging.debug("[" + self.name + ".init " + str(self.initLevel) + "] adding domain %s to symbol %s; Domains now: %s" %(
				str((dependantDomName, dependantDomSize)), self.name, self.domains
			))
		if self.isAutoDom and not self.isPointer:
			alreadyEstablishedDomSizes = [domSize for (domName, domSize) in self.domains]
			logging.debug("[" + self.name + ".init " + str(self.initLevel) + "] Symbol %s is an autoDom symbol: Checking already established domains %s against previous dimensions: %s. dependantDomNameAndSize: %s" %(
				self.name, str(self.domains), str(dimsBeforeReset), str(dependantDomNameAndSize))
			)
			for (domName, domSize) in dimsBeforeReset:
				if domName in self.parallelActiveDims \
				or parallelRegionDomNamesBySize.get(domSize) != None \
				or (len(dimsBeforeReset) <= len(self.domains) and domSize in alreadyEstablishedDomSizes):
					continue
				self.domains.append((domName, domSize))
		self.checkIntegrityOfDomains()

	def loadModuleNodeAttributes(self, moduleNode):
		if self.initLevel < Init.DEPENDANT_ENTRYNODE_ATTRIBUTES_LOADED:
			raise Exception("Symbol %s's routine node attributes are loaded without loading the entry node attributes first."
				%(str(self))
			)
		logging.debug("[" + self.name + ".init " + str(self.initLevel) + "] +++++++++ LOADING MODULE NODE ++++++++++ ")
		self.routineNode = moduleNode #MMU 2015-11-18: $$$ This needs to be commented or rethought
		self.loadTemplateAttributes()
		self.initLevel = max(self.initLevel, Init.ROUTINENODE_ATTRIBUTES_LOADED)
		logging.debug("[" + self.name + ".init " + str(self.initLevel) + "] symbol attributes loaded from module node. Domains at this point: %s. Init Level: %s" %(str(self.domains), str(self.initLevel)))

	def loadRoutineNodeAttributes(self, routineNode, parallelRegionTemplates):
		if self.initLevel < Init.DEPENDANT_ENTRYNODE_ATTRIBUTES_LOADED:
			raise Exception("Symbol %s's routine node attributes are loaded without loading the entry node attributes first."
				%(str(self))
			)
		logging.debug("[" + self.name + ".init " + str(self.initLevel) + "] +++++++++ LOADING ROUTINE NODE ++++++++++ ")
		self.routineNode = routineNode
		#get and check parallelRegionPosition
		routineName = self.nameOfScope
		if not routineName:
			raise Exception("Routine node without name: %s" %(routineNode.toxml()))
		parallelRegionPosition = routineNode.getAttribute("parallelRegionPosition")
		parallelRegionTemplatesUsedForLoading = []
		if parallelRegionPosition and parallelRegionPosition != "":
			self.parallelRegionPosition = parallelRegionPosition
			self.parallelRegionTemplates = parallelRegionTemplates
			if parallelRegionPosition not in ["inside", "outside", "within"]:
				raise Exception("Invalid parallel region position definition ('%s') for routine %s" %(parallelRegionPosition, routineName))
			parallelRegionTemplatesUsedForLoading = parallelRegionTemplates
		self.loadTemplateAttributes(parallelRegionTemplatesUsedForLoading)
		self.initLevel = max(self.initLevel, Init.ROUTINENODE_ATTRIBUTES_LOADED)
		logging.debug("[" + self.name + ".init " + str(self.initLevel) + "] routine node attributes loaded for symbol %s. Domains at this point: %s" %(self.name, str(self.domains)))

	def loadDeclaration(self, paramDeclMatch, patterns, currentRoutineArguments):
		if self.initLevel < Init.TEMPLATE_LOADED:
			raise Exception(
				"Cannot load declaration for %s at init level %s" %(
					str(self),
					self.initLevel
				)
			)

		logging.debug("[" + self.name + ".init " + str(self.initLevel) + "] +++++++++ LOADING DECLARATION ++++++++++ ")

		#   The name used in the declaration pattern is just self.name - so store this as the scoped name for now
		self._nameInScope = self.name

		declarationDirectives, symbolDeclarationStr = purgeFromDeclarationSettings( \
			paramDeclMatch.group(0), \
			[self], \
			patterns, \
			withAndWithoutIntent=False \
		)
		self.declarationPrefix = purgeDimensionAndGetAdjustedLine(declarationDirectives.rstrip() + " " + "::", patterns)

		#   get and check intent                                      #
		intentMatch = patterns.intentPattern.match(paramDeclMatch.group(1))
		newIntent = None
		if intentMatch and intentMatch.group(1).strip() != "":
			newIntent = intentMatch.group(1)
		elif self.name in currentRoutineArguments:
			newIntent = "unspecified" #dummy symbol without specified intent (basically F77 style)
		else:
			newIntent = "local"
		if newIntent not in ["", None, "unspecified"]:
			self.intent = newIntent

		#   check whether this is a pointer
		self.isPointer = self.pointerDeclarationPattern.match(paramDeclMatch.group(0)) != None

		#   look at declaration of symbol and get its                 #
		#   dimensions.                                               #
		dimensionStr, remainder = self.getDimensionStringAndRemainderFromDeclMatch(
			paramDeclMatch,
			patterns.dimensionPattern
		)
		dimensionSizes = [sizeStr.strip() for sizeStr in dimensionStr.split(',') if sizeStr.strip() != ""]
		if self.isAutoDom:
			logging.debug("[" + self.name + ".init " + str(self.initLevel) + "] reordering domains for symbol %s with autoDom option." %(self.name))
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
			logging.debug("[" + self.name + ".init " + str(self.initLevel) + "] Loading dimensions for autoDom, non-pointer symbol %s. Declared dimensions: %s, Known dimension sizes used for parallel regions: %s, Parallel Active Dims: %s, Parallel Inactive Dims: %s" %(
				str(self), str(dimensionSizes), str(self.aggregatedRegionDomSizesByName), str(self.parallelActiveDims), str(self.parallelInactiveDims)
			))
			for dimensionSize in dimensionSizes:
				missingParallelDomain = None
				for domName in self.aggregatedRegionDomNames:
					if not dimensionSize in self.aggregatedRegionDomSizesByName[domName]:
						continue
					#we have found the dimension size that this symbol expects for this domain name. -> use it
					self.aggregatedRegionDomSizesByName[domName] = [dimensionSize]
					if domName in self.parallelActiveDims:
						continue
					missingParallelDomain = domName
					break
				if missingParallelDomain != None:
					logging.debug("[" + self.name + ".init " + str(self.initLevel) + "] Dimension size %s matched to a parallel region but not matched in the domain dependant \
template for symbol %s - automatically inserting it for domain name %s\n"
						%(dimensionSize, self.name, domName)
					)
					self.parallelActiveDims.append(domName)
			self.domains = []
			self.parallelInactiveDims = []
			for parallelDomName in self.parallelActiveDims:
				parallelDomSizes = self.aggregatedRegionDomSizesByName.get(parallelDomName)
				if parallelDomSizes == None or len(parallelDomSizes) == 0:
					raise Exception("No domain size found for domain name %s" %(parallelDomName))
				elif len(parallelDomSizes) > 1:
					raise Exception("There are multiple known dimension sizes for domain %s. Cannot insert domain for autoDom symbol %s. Please use explicit declaration" %(parallelDomName, str(self)))
				self.domains.append((parallelDomName, parallelDomSizes[0]))
			for dimensionSize in dimensionSizes:
				for domName in self.aggregatedRegionDomNames:
					if dimensionSize in self.aggregatedRegionDomSizesByName[domName]:
						break
				else:
					self.parallelInactiveDims.append(dimensionSize)
					self.domains.append(("HF_GENERIC_PARALLEL_INACTIVE_DIM", dimensionSize))
			logging.debug("[" + self.name + ".init " + str(self.initLevel) + "] done loading autoDom dimensions. Parallel Active Dims: %s, Parallel Inactive Dims: %s" %(str(self.parallelActiveDims), str(self.parallelInactiveDims)))

		# at this point we may not go further if the parallel region data
		# has not yet been analyzed.
		if not self.parallelRegionPosition:
			if not self.isPointer:
				self.domains = getReorderedDomainsAccordingToDeclaration(self.domains, dimensionSizes, purgeUndeclared=True)
			self.checkIntegrityOfDomains()
			return

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
				parallelDomSizes = self.aggregatedRegionDomSizesByName.get(parallelDomName)
				if parallelDomSizes == None or len(parallelDomSizes) == 0:
					raise Exception("No domain size found for domain name %s" %(parallelDomName))
				for parallelDomSize in parallelDomSizes:
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
					if len(parallelDomSizes) > 1:
						raise Exception("There are multiple known dimension sizes for domain %s. Cannot insert domain for autoDom symbol %s. Please use explicit declaration" %(parallelDomName, str(self)))
					lastParallelDomainIndex += 1
					self.domains.insert(lastParallelDomainIndex, (parallelDomName, parallelDomSizes[0]))
			if self.parallelRegionPosition == "outside":
				self.domains = [(domName, domSize) for (domName, domSize) in self.domains if not domName in self.parallelActiveDims]
			logging.debug("[" + self.name + ".init " + str(self.initLevel) + "] parallel active dims analysed")

		#   Now match the declared dimensions to those in the         #
		#   'parallelInactive' set, using the declared domain sizes.  #
		#   All should be matched, otherwise throw an error.          #
		#   Insert the dimensions in order of their appearance in     #
		#   the domainDependant template.                             #
		dimensionSizesMatchedInTemplate = []
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
		logging.debug("[" + self.name + ".init " + str(self.initLevel) + "] dependant directive analysed")

		#    Sanity checks                                            #
		if len(self.domains) < len(dimensionSizes):
			raise Exception("Something is wrong with autoDom Symbol %s's declaration: Cannot match its dimension sizes to the parallel regions it is being used in. \
Please make sure to use the same string names for its dimensions both in the parallel region as well as in its declarations -or- declare its dimensions explicitely (without autoDom).\
Declared domain: %s, Domain after init: %s, Parallel dims: %s, Independant dims: %s, \
Parallel region position: %s, Current template: %s"
				%(self.name, str(dimensionSizes), str(self.domains), str(self.parallelActiveDims), str(self.parallelInactiveDims), self.parallelRegionPosition, self.template.toxml())
			)

		if not self.isAutoDom and len(dimensionSizes) != len(dimensionSizesMatchedInTemplate):
			raise Exception("Symbol %s's domainDependant directive does not specify the flag 'autoDom', \
but the @domainDependant specification doesn't match all the declared dimensions. Either use the 'autoDom' attribute or specify \
all dimensions in the @domainDependant specification.\nNumber of declared dimensions: %i (%s); number of template dimensions: %i (%s), \
Parallel region position: %s"
				%(self.name, len(dimensionSizes), str(dimensionSizes), len(dimensionSizesMatchedInTemplate), str(dimensionSizesMatchedInTemplate), self.parallelRegionPosition)
			)
		if not self.isPointer:
			self.domains = getReorderedDomainsAccordingToDeclaration(self.domains, dimensionSizes)
		self.checkIntegrityOfDomains()
		self.initLevel = max(self.initLevel, Init.DECLARATION_LOADED)
		logging.debug("[" + self.name + ".init " + str(self.initLevel) + "] declaration loaded for symbol %s. Domains at this point: %s" %(self.name, str(self.domains)))

	def loadImportInformation(self, importMatch, cgDoc, moduleNode):
		if self.initLevel < Init.TEMPLATE_LOADED:
			raise Exception(
				"Cannot load import information for %s at init level %s" %(
					str(self),
					self.initLevel
				)
			)
		logging.debug("[" + self.name + ".init " + str(self.initLevel) + "] +++++++++ LOADING IMPORT INFORMATION ++++++++++ ")
		sourceModuleName = importMatch.group(1)
		self.sourceModule = sourceModuleName
		if type(self.sourceModule) != str or self.sourceModule == "":
			raise Exception("Invalid module in use statement for symbol %s" %(symbol.name))

		#   The name used in the import pattern is just self.name - so store this as the scoped name for now
		self._nameInScope = self.name

		mapMatch = self.symbolImportMapPattern.match(importMatch.group(0))
		sourceSymbolName = ""
		if mapMatch:
			sourceSymbolName = mapMatch.group(1)
			if sourceSymbolName == "":
				raise Exception("Invalid source symbol in use statement for symbol %s" %(symbol.name))
		if sourceSymbolName == "":
			sourceSymbolName = self.name
		self.sourceSymbol = sourceSymbolName
		if not moduleNode:
			return

		templatesAndEntries = getDomainDependantTemplatesAndEntries(cgDoc, moduleNode)
		informationLoadedFromModule = False
		routineTemplate = self.template
		moduleTemplate = None
		for template, entry in templatesAndEntries:
			dependantName = entry.firstChild.nodeValue
			if sourceSymbolName != "" and dependantName != sourceSymbolName:
				continue
			elif sourceSymbolName == "" and dependantName != self.name:
				continue
			self.loadDomainDependantEntryNodeAttributes(entry, warnOnOverwrite=False)
			moduleTemplate = template
			break
		else:
			return
			#MMU 2015-9-14: This check fails with older CUDA Fortran based implementations where module data wasn't yet supported
			# raise Exception("Symbol %s not found in module information available to Hybrid Fortran. Please use an appropriate @domainDependant specification." %(self.name))
		informationLoadedFromModule = True
		logging.debug(
				"[" + str(self) + ".init " + str(self.initLevel) + "] Loading symbol information for %s imported from %s (import line: '%s')\n\
Current Domains: %s\n" %(
					self.name, self.sourceModule, importMatch.group(0), str(self.domains)
				)
			)
		attributes, domains, declarationPrefix, accPP, domPP = getAttributesDomainsDeclarationPrefixAndMacroNames(moduleTemplate, routineTemplate)
		self.setOptionsFromAttributes(attributes)
		self.loadDeclarationPrefixFromString(declarationPrefix)
		self.loadDomains(domains, self.parallelRegionTemplates if self.parallelRegionTemplates != None else [])
		self.domains = getReorderedDomainsAccordingToDeclaration(self.domains, self.declaredDimensionSizes)
		self.accPPName = accPP
		self.domPPName = domPP
		self.initLevel = max(self.initLevel, Init.DECLARATION_LOADED)
		logging.debug(
				"[" + str(self) + ".init " + str(self.initLevel) + "] Symbol %s's initialization completed using module information.\nDomains found in module: %s; parallel active: %s; parallel inactive: %s\n" %(
					str(self),
					str(domains),
					str(self.parallelActiveDims),
					str(self.parallelInactiveDims)
				)
			)

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
		# MMU 2015-9-13: This check is not compatible with CUDA Fortran version of helper_functions_gpu
		# dimensionCheckForbiddenCharacters = re.match(r'^(?!.*[()]).*', dimensionStr, re.IGNORECASE)
		# if not dimensionCheckForbiddenCharacters:
		#     raise Exception("Forbidden characters found in declaration of symbol %s: %s. Note: Preprocessor functions in domain dependant declarations are not allowed, only simple definitions." \
		#         %(self.name, dimensionStr))
		return dimensionStr, postfix

	def getAdjustedDeclarationLine(self, paramDeclMatch, parallelRegionTemplates, dimensionPattern):
		'''process everything that happens per h90 declaration symbol'''
		prefix = paramDeclMatch.group(1)
		postfix = paramDeclMatch.group(2)

		dimensionStr, postfix = self.getDimensionStringAndRemainderFromDeclMatch(paramDeclMatch, dimensionPattern)
		return prefix + str(self) + postfix

	def getSanitizedDeclarationPrefix(self, purgeList=[]):
		if self.declarationPrefix in [None, ""]:
			raise Exception("Cannot generate declaration prefix for %s" %(str(self)))
		if len(purgeList) == 0:
			purgeList = ['intent', 'public', 'parameter']
		result = self._getPurgedDeclarationPrefix(purgeList)
		kindMatch = self.patterns.declarationKindPattern.match(result)
		if kindMatch:
			result = kindMatch.group(1) + kindMatch.group(2) + kindMatch.group(3)
		return result.strip().lower()

	def getDeclarationLineForAutomaticSymbol(self, purgeList=[], patterns=H90RegExPatterns.Instance(), name_prefix="", use_domain_reordering=True, skip_on_missing_declaration=False):
		logging.debug("[" + self.name + ".init " + str(self.initLevel) + "] Decl.Line.Gen: Purge List: %s, Name Prefix: %s, Domain Reordering: %s, Skip on Missing: %s." %(
			str(purgeList),
			name_prefix,
			str(use_domain_reordering),
			str(skip_on_missing_declaration)
		))
		if skip_on_missing_declaration and (self.declarationPrefix == None or self.declarationPrefix == ""):
			return ""
		declarationPrefix = self.getSanitizedDeclarationPrefix(purgeList)
		return declarationPrefix.strip() + " " + name_prefix + self.domainRepresentation(use_domain_reordering)

	def selectAllRepresentation(self):
		if self.initLevel < Init.ROUTINENODE_ATTRIBUTES_LOADED:
			raise Exception("Symbol %s's selection representation is accessed without loading the routine node attributes first" %(str(self)))

		result = self.nameInScope()
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

		result = self.nameInScope()
		if len(self.domains) == 0:
			return result
		needsAdditionalClosingBracket = False
		result += "("
		domPP, isExplicit = self.domPP()
		if domPP != "" and ((isExplicit and self.activeDomainsMatchSpecification) or self.numOfParallelDomains != 0):
			#$$$ we need to include the template here to make pointers compatible with templating
			needsAdditionalClosingBracket = True
			result += domPP + "("
		for index, domain in enumerate(self.domains):
			if index != 0:
				result = result + ","
			dimSize = domain[1]
			if dimSize == ":":
				raise Exception("Cannot generate allocation call for symbol %s on the device - one or more dimension sizes are unknown at this point. \
Please specify the domains and their sizes with domName and domSize attributes in the corresponding @domainDependant directive." %(self.name))
			result += dimSize
		if needsAdditionalClosingBracket:
			result += ")"
		result += ")"
		return result

	def domainRepresentation(self, use_domain_reordering=True):
		name = self.nameInScope(
			useDeviceVersionIfAvailable=len(self.domains) > 0
		)
		result = name
		if len(self.domains) == 0:
			return result
		try:
			needsAdditionalClosingBracket = False
			domPP, isExplicit = self.domPP()
			if use_domain_reordering and domPP != "" \
			and (isExplicit or self.numOfParallelDomains > 0) \
			and self.activeDomainsMatchSpecification:
				result = result + "(" + domPP + "("
				needsAdditionalClosingBracket = True
			else:
				result = result + "("
			for i in range(len(self.domains)):
				if i != 0:
					result += ","
				if self.isPointer:
					result += ":"
				else:
					(domName, domSize) = self.domains[i]
					result += domSize.strip()
			if needsAdditionalClosingBracket:
				result = result + "))"
			else:
				result = result + ")"
		except Exception as e:
			return "%s{%s}" %(name, str(self.domains))
		return result

	def totalArrayLength(self):
		result = ""
		for i in range(len(self.domains)):
			if i != 0:
				result += " * "
			(domName, domSize) = self.domains[i]
			sizeParts = domSize.split(':')
			if len(sizeParts) == 1:
				result += sizeParts[0]
			elif len(sizeParts) == 2:
				result += "(%s - %s + 1)" %(sizeParts[1], sizeParts[0])
			else:
				raise Exception("invalid domain size for symbol %s: %s" %(self.name, domSize))

		return result

	def accessRepresentation(self, parallelIterators, offsets, parallelRegionNode, use_domain_reordering=True, inside_subroutine_call=False):
		def getIterators(domains, parallelIterators, offsets):
			iterators = []
			nextOffsetIndex = 0
			if len(parallelIterators) == 0 and len(offsets) == 0:
				return iterators
			for i in range(len(domains)):
				if len(parallelIterators) == 0 and len(offsets) == len(domains):
					iterators.append(str(offsets[i]))
					continue
				elif len(parallelIterators) == 0 \
				and len(offsets) == len(domains) - self.numOfParallelDomains \
				and i < self.numOfParallelDomains:
					iterators.append(":")
					continue
				elif len(parallelIterators) == 0 \
				and len(offsets) == len(domains) - self.numOfParallelDomains \
				and i >= self.numOfParallelDomains:
					iterators.append(str(offsets[i - self.numOfParallelDomains]))
					continue

				#if we reach this there are parallel iterators specified.
				if len(offsets) == len(domains):
					iterators.append(str(offsets[nextOffsetIndex]))
					nextOffsetIndex += 1
				elif domains[i][0] in parallelIterators:
					iterators.append(str(domains[i][0]))
				elif nextOffsetIndex < len(offsets):
					iterators.append(str(offsets[nextOffsetIndex]))
					nextOffsetIndex += 1
				elif len(offsets) + len(parallelIterators) == len(domains) and i < len(parallelIterators):
					iterators.append(str(parallelIterators[i]))
				elif len(offsets) + len(parallelIterators) == len(domains):
					iterators.append(str(offsets[i - len(parallelIterators)]))
				else:
					raise Exception("Cannot generate access representation for symbol %s: Unknown parallel iterators specified (%s) or not enough offsets (%s)."
						%(str(self), str(parallelIterators), str(offsets))
					)
			if inside_subroutine_call and all([iterator == ':' for iterator in iterators]):
				return [] #working around a problem in PGI 15.1: Inliner bails out in certain situations (module test kernel 3+4) if arrays are passed in like a(:,:,:).
			return [iterator.strip().replace(" ", "") for iterator in iterators]

		logging.debug("[" + self.name + ".init " + str(self.initLevel) + "] producing access representation for symbol %s; parallel iterators: %s, offsets: %s" %(self.name, str(parallelIterators), str(offsets)))

		if self.initLevel < Init.ROUTINENODE_ATTRIBUTES_LOADED:
			logging.debug("[" + self.name + ".init " + str(self.initLevel) + "] only returning name since routine attributes haven't been loaded yet.")
			return self.name

		if len(parallelIterators) == 0 \
		and len(offsets) != 0 \
		and len(offsets) != len(self.domains) - self.numOfParallelDomains \
		and len(offsets) != len(self.domains):
			raise Exception("Unexpected number of offsets specified for symbol %s; Offsets: %s, Expected domains: %s" \
				%(self.name, offsets, self.domains))
		if len(parallelIterators) != 0 \
		and len(offsets) + len(parallelIterators) != len(self.domains) \
		and len(offsets) != len(self.domains):
			raise Exception("Unexpected number of offsets and iterators specified for symbol %s; Offsets: %s, Iterators: %s, Expected domains: %s" \
				%(self.name, offsets, parallelIterators, self.domains))

		result = ""
		symbolNameUsedInAccessor = None
		if (not self.isUsingDevicePostfix and len(offsets) == len(self.domains) and not all([offset == ':' for offset in offsets]))\
		or (self.intent == "in" and len(offsets) == len(self.domains) and not any([offset == ':' for offset in offsets])):
			symbolNameUsedInAccessor = self.nameInScope(useDeviceVersionIfAvailable=False) #not on device or scalar accesses to symbol that can't change
		elif self.isUsingDevicePostfix and len(offsets) > 0 and any([offset == ':' for offset in offsets]) and not all([offset == ':' for offset in offsets]):
			raise Exception("Cannot reshape the array %s at this point, it needs to be accessed either at a single value or for the entire array; offsets: %s" %(self, offsets))
		else:
			symbolNameUsedInAccessor = self.nameInScope()
		result += symbolNameUsedInAccessor

		if len(self.domains) == 0:
			logging.debug("[" + self.name + ".init " + str(self.initLevel) + "] Symbol has 0 domains - only returning name.")
			return result
		iterators = getIterators(self.domains, parallelIterators, offsets)
		if len(iterators) == 0:
			logging.debug("[" + self.name + ".init " + str(self.initLevel) + "] No iterators have been determined - only returning name.")
			return result

		needsAdditionalClosingBracket = False
		result += "( " #we add a space here so there is a higher change that the line can be broken up. iterators with preprocessor macros can take a lot of space.
		accPP, accPPIsExplicit = self.accPP()
		logging.debug("[" + self.name + ".init " + str(self.initLevel) + "] accPP Macro: %s, Explicit Macro: %s, Active Domains matching domain dependant template: %s, Number of Parallel Domains: %i\
Currently loaded template: %s\n" %(
				accPP, str(accPPIsExplicit), self.activeDomainsMatchSpecification, self.numOfParallelDomains, self.template.toxml() if self.template != None else "None"
			))
		if use_domain_reordering and accPP != "" \
		and (accPPIsExplicit or self.numOfParallelDomains > 0) \
		and self.activeDomainsMatchSpecification:
			needsAdditionalClosingBracket = True
			if not accPPIsExplicit and parallelRegionNode:
				template = getTemplate(parallelRegionNode)
				if template != '':
					accPP += "_" + template
			result += accPP + "("
		result += ",".join(iterators)

		if needsAdditionalClosingBracket:
			result = result + ")"
		result = result + " )" #same reason for the spacing as for the opening bracket.
		return result

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
			return domPPEntries[0], True

		if self.domPPName not in ['', None]:
			return self.domPPName, True

		if self.isAutoDom:
			numOfDimensions = len(self.domains)
			domPPName = ""
			if numOfDimensions < 3:
				domPPName = ""
			elif numOfDimensions == 3:
				domPPName = "DOM"
			else:
				domPPName = "DOM%i" %(numOfDimensions)
			return domPPName, False
		else:
			return "", False


	def accPP(self):
		accPPEntries = self.getTemplateEntryNodeValues("accPP")
		if accPPEntries and len(accPPEntries) > 0:
			return accPPEntries[0], True

		if self.accPPName not in ['', None]:
			return self.accPPName, True

		if self.isAutoDom:
			numOfDimensions = len(self.domains)
			accPPName = ""
			if numOfDimensions < 3:
				accPPName = ""
			elif numOfDimensions == 3:
				accPPName = "AT"
			else:
				accPPName = "AT%i" %(numOfDimensions)
			return accPPName, False
		else:
			return "", False

class ImplicitForeignModuleSymbol(Symbol):
	def __init__(self, sourceModule, nameInScope, sourceSymbol, template=None):
		Symbol.__init__(self, nameInScope, template)
		self._nameInScope = nameInScope
		self.sourceModule = sourceModule
		self.sourceSymbol = sourceSymbol

class FrameworkArray(Symbol):
	def __init__(self, name, declarationPrefix, domains, isOnDevice):
		if not name or name == "":
			raise Exception("Name required for initializing framework array")
		if not declarationPrefix or declarationPrefix == "":
			raise Exception("Declaration prefix required for initializing framework array")
		if len(domains) != 1:
			raise Exception("Currently unsupported non-1D-array specified as framework array")
		Symbol.__init__(self, name)
		self.domains = domains
		self.isMatched = True
		self.isOnDevice = isOnDevice
		self.isConstant = True
		self.declarationPrefix = declarationPrefix
		self._declarationTypeOverride = DeclarationType.FRAMEWORK_ARRAY