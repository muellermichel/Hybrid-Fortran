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

#   Boxes = Symbol States
#   X -> Transition Texts = Methods being called by parser
#   Other Texts = Helper Functions
#
#                                        +----------------+
#                                        | NOTHING_LOADED |
#                                        +------+---------+
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

def getReorderedDomainsAccordingToDeclaration(domains, dimensionSizesInDeclaration):
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

    if len(domains) != len(dimensionSizesInDeclaration) or len(domains) == 0:
        return domains
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
    isModuleSymbol = False
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
    parallelRegionTemplates = None

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
        self.isModuleSymbol = False
        self.parallelRegionTemplates = None

        self.isPresent = False
        self.isAutoDom = False
        self.isToBeTransfered = False
        self.isCompacted = False
        attributes = getAttributes(self.template)
        self.setOptionsFromAttributes(attributes)

    def __repr__(self):
        name = self.name
        if self.isAutomatic:
            name = self.automaticName()
        elif len(self.domains) > 0:
            name = self.deviceName()
        result = name
        if len(self.domains) == 0:
            return result
        try:
            needsAdditionalClosingBracket = False
            domPP, isExplicit = self.domPP()
            if domPP != "" and ((isExplicit and self.activeDomainsSameAsTemplate) or self.parallelRegionPosition != "outside"):
                result = result + "(" + domPP + "("
                needsAdditionalClosingBracket = True
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
            if needsAdditionalClosingBracket:
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

    @property
    def activeDomainsSameAsTemplate(self):
        if not self.template:
            return False
        templateDomains = getDomNameAndSize(self.template)
        if len(self.domains) == len(templateDomains):
            return True
        return False

    def setOptionsFromAttributes(self, attributes):
        if "present" in attributes:
            self.isPresent = True
        if "autoDom" in attributes:
            self.isAutoDom = True
        if "transferHere" in attributes:
            if self.isPresent:
                raise Exception("Symbol %s has contradicting attributes 'transferHere' and 'present'" %(self))
            self.isToBeTransfered = True

    def storeDomainDependantEntryNodeAttributes(self, domainDependantEntryNode):
        if self.debugPrint:
            sys.stderr.write("storing symbol attributes for %s. Init Level: %s\n" %(str(self), str(self.initLevel)))
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
        if warnOnOverwrite and self.initLevel > Init.NOTHING_LOADED:
            sys.stderr.write("WARNING: symbol %s's entry node attributes are loaded when the initialization level has already advanced further\n" \
                %(str(self))
            )

        self.intent = domainDependantEntryNode.getAttribute("intent")
        self.declarationPrefix = domainDependantEntryNode.getAttribute("declarationPrefix")
        self.sourceModule = domainDependantEntryNode.getAttribute("sourceModule")
        self.sourceSymbol = domainDependantEntryNode.getAttribute("sourceSymbol")
        self.isPointer = domainDependantEntryNode.getAttribute("isPointer") == "yes"
        self.declaredDimensionSizes = domainDependantEntryNode.getAttribute("declaredDimensionSizes").split(",")
        if len(self.declaredDimensionSizes) > 0:
            self.domains = []
        for dimSize in self.declaredDimensionSizes:
            if dimSize.strip() != "":
                self.domains.append(('HF_GENERIC_DIM', dimSize))
        self.initLevel = max(self.initLevel, Init.DEPENDANT_ENTRYNODE_ATTRIBUTES_LOADED)

    def checkIntegrityOfDomains(self):
        for domain in self.domains:
            if not hasattr(domain, '__iter__'):
                raise Exception("Invalid definition of domain in symbol %s: %s" %(self.name, str(domain)))

    def loadTemplateAttributes(self, parallelRegionTemplates=[]):
        dependantDomNameAndSize = getDomNameAndSize(self.template)
        declarationPrefixFromTemplate = getDeclarationPrefix(self.template)
        self.loadDeclarationPrefixFromString(declarationPrefixFromTemplate)
        if not self.isModuleSymbol and (not parallelRegionTemplates or len(parallelRegionTemplates) == 0):
            sys.stderr.write("WARNING: No active parallel region found, in subroutine %s where dependants are defined\n" %(self.routineNode.getAttribute("name")))
            return
        self.loadDomains(dependantDomNameAndSize, parallelRegionTemplates)

    def loadDeclarationPrefixFromString(self, declarationPrefixFromTemplate):
        if declarationPrefixFromTemplate != None and declarationPrefixFromTemplate.strip() != "":
            self.declarationPrefix = declarationPrefixFromTemplate

    def loadDomains(self, dependantDomNameAndSize, parallelRegionTemplates=[]):
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
        for parallelRegionTemplate in parallelRegionTemplates:
            regionDomNameAndSize = getDomNameAndSize(parallelRegionTemplate)
            for (regionDomName, regionDomSize) in regionDomNameAndSize:
                if regionDomName in dependantDomSizeByName.keys() and regionDomName not in self.parallelActiveDims:
                    self.parallelActiveDims.append(regionDomName)
                #The same domain name can sometimes have different domain sizes used in different parallel regions, so we build up a list of these sizes.
                if not regionDomName in self.aggregatedRegionDomSizesByName:
                    self.aggregatedRegionDomSizesByName[regionDomName] = [regionDomSize]
                else:
                    self.aggregatedRegionDomSizesByName[regionDomName].append(regionDomSize)
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
            alreadyEstablishedDomSizes = [domSize for (domName, domSize) in self.domains]
            for (domName, domSize) in dimsBeforeReset:
                if len(dimsBeforeReset) == len(self.domains) and domSize in alreadyEstablishedDomSizes:
                    continue
                self.domains.append((domName, domSize))
        self.checkIntegrityOfDomains()
        if self.debugPrint:
            sys.stderr.write("Domains loaded from callgraph information for symbol %s. Parallel active: %s. Parallel Inactive: %s." %(
                str(self), str(self.parallelActiveDims), str(self.parallelInactiveDims)
            ))

    def loadModuleNodeAttributes(self, moduleNode):
        if self.initLevel < Init.DEPENDANT_ENTRYNODE_ATTRIBUTES_LOADED:
            raise Exception("Symbol %s's routine node attributes are loaded without loading the entry node attributes first."
                %(str(self))
            )
        if self.initLevel > Init.DEPENDANT_ENTRYNODE_ATTRIBUTES_LOADED:
            sys.stderr.write("WARNING: symbol %s's routine node attributes are loaded when the initialization level has already advanced further\n" \
                %(str(self))
            )
        self.routineNode = moduleNode
        self.loadTemplateAttributes()
        self.initLevel = max(self.initLevel, Init.ROUTINENODE_ATTRIBUTES_LOADED)
        if self.debugPrint:
            sys.stderr.write("symbol attributes loaded from module node for %s. Domains at this point: %s. Init Level: %s\n" %(str(self), str(self.domains), str(self.initLevel)))

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
        self.parallelRegionTemplates = parallelRegionTemplates
        self.loadTemplateAttributes(parallelRegionTemplates)
        self.initLevel = max(self.initLevel, Init.ROUTINENODE_ATTRIBUTES_LOADED)
        if self.debugPrint:
            sys.stderr.write("routine node attributes loaded for symbol %s. Domains at this point: %s\n" %(self.name, str(self.domains)))

    def loadDeclaration(self, paramDeclMatch, patterns):
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
            if self.debugPrint:
                sys.stderr.write("Loading dimensions for autoDom, non-pointer symbol %s. Declared dimensions: %s, Known dimension sizes used for parallel regions: %s\n" %(
                    str(self), str(dimensionSizes), str(self.aggregatedRegionDomSizesByName)
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
                    if self.debugPrint:
                        sys.stderr.write("Dimension size %s matched to a parallel region but not matched in the domain dependant \
template for symbol %s - automatically inserting it for domain name %s\n"
                            %(dimensionSize, self.name, domName)
                        )
                    self.parallelActiveDims.append(domName)
            self.domains = []
            for parallelDomName in self.parallelActiveDims:
                parallelDomSizes = self.aggregatedRegionDomSizesByName.get(parallelDomName)
                if parallelDomSizes == None or len(parallelDomSizes) == 0:
                    raise Exception("Unexpected Error: No domain size found for domain name %s" %(parallelDomName))
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

        # at this point we may not go further if the parallel region data
        # has not yet been analyzed.
        if not self.parallelRegionPosition:
            if not self.isPointer:
                self.domains = getReorderedDomainsAccordingToDeclaration(self.domains, dimensionSizes)
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
                    raise Exception("Unexpected Error: No domain size found for domain name %s" %(parallelDomName))
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
        if self.debugPrint:
            sys.stderr.write("declaration loaded for symbol %s. Domains at this point: %s\n" %(self.name, str(self.domains)))

    def loadImportInformation(self, importMatch, cgDoc, moduleNode):
        if self.initLevel > Init.ROUTINENODE_ATTRIBUTES_LOADED:
            sys.stderr.write("WARNING: symbol %s's import information is loaded when the initialization level has already advanced further.\n" \
                %(str(self))
            )

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
            raise Exception("Symbol %s not found in module information available to Hybrid Fortran. Please use an appropriate @domainDependant specification.")
        informationLoadedFromModule = True
        if self.debugPrint:
            sys.stderr.write(
                "Loading symbol information for %s imported from %s\n\
Procedure @domainDependant specification: %s\n\
Imported specification: %s\n\
Current Domains: %s\n" %(
                    self.name, sourceModuleName, routineTemplate.toxml(), moduleTemplate.toxml(), str(self.domains)
                )
            )
        attributes, domains, declarationPrefix = getAttributesDomainsAndDeclarationPrefixFromModuleTemplateAndProcedureTemplateForProcedure(moduleTemplate, routineTemplate)
        self.setOptionsFromAttributes(attributes)
        self.loadDeclarationPrefixFromString(declarationPrefix)
        self.loadDomains(domains, self.parallelRegionTemplates if self.parallelRegionTemplates != None else [])
        self.domains = getReorderedDomainsAccordingToDeclaration(self.domains, self.declaredDimensionSizes)
        self.initLevel = max(self.initLevel, Init.DECLARATION_LOADED)
        if self.debugPrint:
            sys.stderr.write(
                "Symbol %s's initialization completed using module information.\nDomains found in module: %s.\nParallel Region Templates: %s\n" %(
                    str(self),
                    str(domains),
                    str([template.toxml() for template in self.parallelRegionTemplates]) if self.parallelRegionTemplates != None else "None"
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
Please either use an @domainDependant specification in the imported module's module scope OR \
specify the type like in a Fortran 90 declaration line using a @domainDependant {declarationPrefix([TYPE DECLARATION])} directive within the current subroutine.\n\n\
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
        needsAdditionalClosingBracket = False
        result += "("
        domPP, isExplicit = self.domPP()
        if domPP != "" and ((isExplicit and self.activeDomainsSameAsTemplate) or self.numOfParallelDomains != 0):
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

    def accessRepresentation(self, parallelIterators, offsets, parallelRegionNode):
        if self.debugPrint:
            sys.stderr.write("producing access representation for symbol %s; parallel iterators: %s, offsets: %s\n" %(self.name, str(parallelIterators), str(offsets)))

        if self.initLevel < Init.ROUTINENODE_ATTRIBUTES_LOADED:
            if self.debugPrint:
                sys.stderr.write("only returning name since routine attributes haven't been loaded yet.\n")
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
            if self.debugPrint:
                sys.stderr.write("Symbol has 0 domains - only returning name.\n")
            return result
        needsAdditionalClosingBracket = False
        result = result + "("
        accPP, accPPIsExplicit = self.accPP()
        if (not accPPIsExplicit or not self.activeDomainsSameAsTemplate) and self.numOfParallelDomains != 0 and accPP != "":
            if parallelRegionNode:
                template = getTemplate(parallelRegionNode)
                if template != '':
                    accPP += "_" + template
            result = result + accPP + "("
            needsAdditionalClosingBracket = True
        elif accPPIsExplicit and self.activeDomainsSameAsTemplate and accPP != "":
            result = result + accPP + "("
            needsAdditionalClosingBracket = True
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
            if len(offsets) == len(self.domains):
                result += offsets[nextOffsetIndex]
                nextOffsetIndex += 1
            elif self.domains[i][0] in parallelIterators:
                result += self.domains[i][0]
            elif nextOffsetIndex < len(offsets):
                result += offsets[nextOffsetIndex]
                nextOffsetIndex += 1
            elif len(offsets) + len(parallelIterators) == len(self.domains) and i < len(parallelIterators):
                result += parallelIterators[i]
            elif len(offsets) + len(parallelIterators) == len(self.domains):
                result += offsets[i - len(parallelIterators)]
            else:
                raise Exception("Cannot generate access representation for symbol %s: Unknown parallel iterators specified (%s) or not enough offsets (%s)."
                    %(str(self), str(parallelIterators), str(offsets))
                )

        if needsAdditionalClosingBracket:
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
            return domPPEntries[0], True

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


