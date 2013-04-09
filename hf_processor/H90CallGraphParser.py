#!/usr/bin/python
# -*- coding: UTF-8 -*-

# Copyright (C) 2013 Michel Müller, Rikagaku Kenkyuujo (RIKEN)

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
#  Procedure        H90CallGraphParser.py                              #
#  Comment          Generates a Fortran callgraph in xml format        #
#                   including parallel region annotations.             #
#                   For parsing it uses a combination of               #
#                   a finite state machine and regex (per line)        #
#  Date             2012/07/27                                         #
#  Author           Michel Müller (AOKI Laboratory)                    #
#**********************************************************************#


from xml.dom.minidom import Document
from DomHelper import *
from GeneralHelper import BracketAnalyzer, findRightMostOccurrenceNotInsideQuotes, stripWhitespace
from H90Symbol import *
from H90RegExPatterns import H90RegExPatterns
import os
import sys
import fileinput
import re
import uuid
import pdb
import traceback

class FortranCodeSanitizer:
    currNumOfTabs = 0
    wasLastLineEmpty = False
    tabIncreasingPattern = None
    tabDecreasingPattern = None
    endifOrModulePattern = None
    endSubroutinePattern = None
    commentedPattern = None
    preprocessorPattern = None

    def __init__(self):
        self.tabIncreasingPattern = re.compile(r'.*?(?:select|do|subroutine|function)\s.*', re.IGNORECASE)
        self.endifOrModulePattern = re.compile(r'end\s*(if|module).*', re.IGNORECASE)
        self.endSubroutinePattern = re.compile(r'end\s*subroutine.*', re.IGNORECASE)
        self.tabDecreasingPattern = re.compile(r'(end\s|enddo).*', re.IGNORECASE)
        self.commentedPattern = re.compile(r'\s*\!', re.IGNORECASE)
        self.preprocessorPattern = re.compile(r'\s*\#', re.IGNORECASE)
        self.currNumOfTabs = 0

    def sanitizeLines(self, line, toBeCommented=False, howManyCharsPerLine=132, commentChar="!"):
        strippedRawLine = line.strip()
        if strippedRawLine == "" and self.wasLastLineEmpty:
            return ""
        if strippedRawLine == "":
            self.wasLastLineEmpty = True
            return "\n"

        self.wasLastLineEmpty = False
        codeLines = strippedRawLine.split("\n")
        sanitizedCodeLines = []
        lineSep = " &"

        # ----------- break up line to get valid Fortran lenghts ------ #
        for codeLine in codeLines:
            if len(codeLine) <= howManyCharsPerLine:
                sanitizedCodeLines.append(codeLine)
                continue
            currLine = codeLine
            while len(currLine) > howManyCharsPerLine - len(lineSep):
                if commentChar in currLine:
                    commentPos = currLine.find(commentChar)
                    if commentPos <= howManyCharsPerLine:
                        break
                #find a blank that's NOT within a quoted string
                blankPos = findRightMostOccurrenceNotInsideQuotes(' ', currLine, rightStartAt=howManyCharsPerLine - len(lineSep))
                if blankPos < 1:
                    #blank not found or at beginning of line
                    #-> bail out in order to avoid infinite loop - just keep the line as it was.
                    raise Exception("The following line could not be broken up for Fortran compatibility - no suitable spaces found: %s" %(currLine))
                sanitizedCodeLines.append(currLine[:blankPos] + lineSep)
                currLine = currLine[blankPos:]
            if toBeCommented:
                currLine = commentChar + " " + currLine
            sanitizedCodeLines.append(currLine)

        # ----------- re indent codelines ----------------------------- #
        # ----------- and strip whitespace ---------------------------- #
        # ----------- and remove consecutive empty lines -------------- #
        tabbedCodeLines = []
        for codeLine in sanitizedCodeLines:
            strippedLine = codeLine.strip()
            if strippedLine == "" and self.wasLastLineEmpty:
                continue
            if strippedLine == "":
                self.wasLastLineEmpty = True
                tabbedCodeLines.append("")
                continue
            self.wasLastLineEmpty = False
            if self.commentedPattern.match(strippedLine):
                tabbedCodeLines.append(strippedLine)
            elif self.preprocessorPattern.match(strippedLine):
                #note: ifort's preprocessor can't handle preprocessor lines with leading whitespace -.-
                #=> catch this case and strip any whitespace.
                tabbedCodeLines.append(strippedLine)
            elif self.endSubroutinePattern.match(strippedLine):
                #note: this is being done in order to 'heal' the tabbing as soon as the subroutine
                #is finished - otherwise there are some edge cases which may propagate.
                self.currNumOfTabs = 0
                tabbedCodeLines.append(strippedLine)
            elif self.endifOrModulePattern.match(strippedLine):
                tabbedCodeLines.append(self.currNumOfTabs * "\t" + strippedLine)
            elif self.tabDecreasingPattern.match(strippedLine):
                if self.currNumOfTabs > 0:
                    self.currNumOfTabs = self.currNumOfTabs - 1
                else:
                    self.currNumOfTabs = 0
                tabbedCodeLines.append(self.currNumOfTabs * "\t" + strippedLine)
            elif self.tabIncreasingPattern.match(strippedLine):
                tabbedCodeLines.append(self.currNumOfTabs * "\t" + strippedLine)
                self.currNumOfTabs = self.currNumOfTabs + 1
            else:
                tabbedCodeLines.append(self.currNumOfTabs * "\t" + strippedLine)

        return "\n".join(tabbedCodeLines) + "\n"

class H90CallGraphParser(object):
    '''An imperative python parser for h90 sourcefiles, based on a finite state machine and regex'''
    '''This class is intended as an abstract class to be inherited and doesn't do anything useful by itself'''
    '''A minimal implementation of this class implements one or more process*Match routines'''
    debugPrint = False
    state = 'none'
    stateBeforeCall = "undefined"
    currSubprocName = None
    currBracketAnalyzer = None
    currCalleeName = None
    patterns = None
    currentLine = None

    def __init__(self):
        self.patterns = H90RegExPatterns()
        self.state = "none"
        self.stateBeforeCall = "undefined"
        self.currCalleeName = None

        super(H90CallGraphParser, self).__init__()

    def purgeTrailingCommentsAndGetAdjustedLine(self, line):
        commentPos = findRightMostOccurrenceNotInsideQuotes("!", line)
        if commentPos < 0:
            return line
        return line[:commentPos]

    def processCallMatch(self, subProcCallMatch):
        if (not subProcCallMatch.group(1) or subProcCallMatch.group(1) == ''):
            raise Exception("subprocedure call without matching subprocedure name")
        self.currBracketAnalyzer = BracketAnalyzer('(', ')')
        level = self.currBracketAnalyzer.currLevelAfterString(subProcCallMatch.group(0))
        self.currCalleeName = subProcCallMatch.group(1)
        if level > 0:
            self.stateBeforeCall = self.state
            self.state = "inside_subroutine_call"
        else:
            self.currBracketAnalyzer = None
        return

    def processProcBeginMatch(self, subProcBeginMatch):
        if self.debugPrint:
            sys.stderr.write('entering %s\n' %(subProcBeginMatch.group(1)))
        return

    def processProcEndMatch(self, subProcEndMatch):
        if self.debugPrint:
            sys.stderr.write('exiting subprocedure\n')
        return

    def processParallelRegionMatch(self, parallelRegionMatch):
        return

    def processParallelRegionEndMatch(self, parallelRegionEndMatch):
        return

    def processDomainDependantMatch(self, domainDependantMatch):
        return

    def processDomainDependantEndMatch(self, domainDependantEndMatch):
        return

    def processNoMatch(self):
        return

    def processSpecificationBeginning(self):
        return

    def processNoneState(self, line):
        specificationsBeginHere = False
        subProcBeginMatch = self.patterns.subprocBeginPattern.match(str(line))
        if self.currBracketAnalyzer:
            level = self.currBracketAnalyzer.currLevelAfterString(line)
            if level == 0:
                self.state ='inside_declarations'
                self.currBracketAnalyzer = None
                specificationsBeginHere = True
            self.processNoMatch()

        elif subProcBeginMatch:
            if (not subProcBeginMatch.group(1) or subProcBeginMatch.group(1) == ''):
                raise Exception("subprocedure begin without matching subprocedure name")
            self.currSubprocName = subProcBeginMatch.group(1)
            self.currBracketAnalyzer = BracketAnalyzer('(', ')')
            level = self.currBracketAnalyzer.currLevelAfterString(line)
            if level == 0:
                self.state ='inside_declarations'
                self.currBracketAnalyzer = None
                specificationsBeginHere = True
            self.processProcBeginMatch(subProcBeginMatch)

        elif (self.patterns.subprocEndPattern.match(str(line))):
            raise Exception("end subprocedure without matching begin subprocedure")

        else:
            self.processNoMatch()

        if specificationsBeginHere:
            self.processSpecificationBeginning()

    def processInsideBranch(self, line):
        return

    def processInsideIgnore(self, line):
        return

    def processInsideSubroutineCall(self, line):
        level = self.currBracketAnalyzer.currLevelAfterString(line)
        if level == 0:
            self.currBracketAnalyzer = None
            self.state = self.stateBeforeCall
            self.stateBeforeCall = "undefined"

    def processInsideDeclarationsState(self, line):
        subProcCallMatch = self.patterns.subprocCallPattern.match(str(line))
        parallelRegionMatch = self.patterns.parallelRegionPattern.match(str(line))
        domainDependantMatch = self.patterns.domainDependantPattern.match(str(line))
        subProcEndMatch = self.patterns.subprocEndPattern.match(str(line))

        if (domainDependantMatch):
            self.processDomainDependantMatch(domainDependantMatch)
            self.state = 'inside_domainDependantRegion'
        elif subProcCallMatch:
            self.processCallMatch(subProcCallMatch)
        elif (subProcEndMatch):
            self.processProcEndMatch(subProcEndMatch)
            self.state = 'none'
            self.currBracketAnalyzer = None
            self.currSubprocName = None
        elif (parallelRegionMatch):
            raise Exception("parallel region without parallel dependants")
        elif (self.patterns.subprocBeginPattern.match(str(line))):
            raise Exception("subprocedure within subprocedure not allowed")
        return

    def processInsideSubroutineBodyState(self, line):
        #note: Branches (@if statements) are ignored here, we want to keep analyzing their statements for callgraphs.
        domainDependantMatch = self.patterns.domainDependantPattern.match(str(line))
        subProcCallMatch = self.patterns.subprocCallPattern.match(str(line))
        parallelRegionMatch = self.patterns.parallelRegionPattern.match(str(line))
        domainDependantMatch = self.patterns.domainDependantPattern.match(str(line))
        subProcEndMatch = self.patterns.subprocEndPattern.match(str(line))

        if domainDependantMatch:
            self.processDomainDependantMatch(domainDependantMatch)
            self.state = 'inside_domainDependantRegion'
        elif subProcCallMatch:
            self.processCallMatch(subProcCallMatch)
        elif subProcEndMatch:
            self.processProcEndMatch(subProcEndMatch)
            self.state = 'none'
            self.currBracketAnalyzer = None
            self.currSubprocName = None
        elif parallelRegionMatch:
            self.processParallelRegionMatch(parallelRegionMatch)
            self.state = 'inside_parallelRegion'
        elif self.patterns.subprocBeginPattern.match(str(line)):
            raise Exception("subprocedure within subprocedure not allowed")
        return

    def processInsideParallelRegionState(self, line):
        subProcCallMatch = self.patterns.subprocCallPattern.match(str(line))
        parallelRegionEndMatch = self.patterns.parallelRegionEndPattern.match(str(line))
        if subProcCallMatch:
            self.processCallMatch(subProcCallMatch)
        elif (parallelRegionEndMatch):
            self.processParallelRegionEndMatch(parallelRegionEndMatch)
            self.state = "inside_subroutine_body"
        elif (self.patterns.parallelRegionPattern.match(str(line))):
            raise Exception("parallelRegion within parallelRegion not allowed")
        elif (self.patterns.subprocEndPattern.match(str(line))):
            raise Exception("subprocedure end before @end parallelRegion")
        elif (self.patterns.subprocBeginPattern.match(str(line))):
            raise Exception("subprocedure within subprocedure not allowed")
        else:
            self.processNoMatch()

    def processInsideDomainDependantRegionState(self, line):
        if self.patterns.domainDependantEndPattern.match(str(line)):
            self.state = "inside_subroutine_body"
        elif self.patterns.subprocCallPattern.match(str(line)):
            raise Exception("subprocedure call within domainDependants not allowed")
        elif (self.patterns.parallelRegionEndPattern.match(str(line)) or self.patterns.parallelRegionPattern.match(str(line))):
            raise Exception("parallelRegion within domainDependants not allowed")
        elif (self.patterns.subprocEndPattern.match(str(line))):
            raise Exception("subprocedure end before @end domainDependant")
        elif (self.patterns.subprocBeginPattern.match(str(line))):
            raise Exception("subprocedure within subprocedure not allowed")
        return

    def processUndefinedState(self, line):
        raise Exception("unexpected undefined parser state: %s" %(self.state))

    def processLine(self, line):
        #define the states and their respective handlers
        stateSwitch = {
           'none': self.processNoneState,
           'inside_declarations': self.processInsideDeclarationsState,
           'inside_parallelRegion': self.processInsideParallelRegionState,
           'inside_domainDependantRegion': self.processInsideDomainDependantRegionState,
           'inside_subroutine_body':self.processInsideSubroutineBodyState,
           'inside_subroutine_call':self.processInsideSubroutineCall,
           'inside_branch':self.processInsideBranch,
           'inside_ignore':self.processInsideIgnore
         }
        #exclude commented lines from analysis
        if (self.patterns.commentedPattern.match(str(line))):
            self.currentLine = line
            return

        #remove trailing comments
        self.currentLine = self.purgeTrailingCommentsAndGetAdjustedLine(str(line))

        #analyse this line. handle the line according to current parser state.
        stateSwitch.get(self.state, self.processUndefinedState)(self.currentLine)
        if not self.state == "inside_subroutine_call":
            self.currCalleeName = None

    def processFile(self, fileName):
        lineNo = 1
        for line in fileinput.input([fileName]):
            try:
                # if lineNo == 70:
                #     pdb.set_trace()
                self.processLine(line)
            except Exception, e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                sys.stderr.write('Error when parsing file %s on line %i; Print of line:\n%s\n%s\n'
                    %(str(fileName), lineNo, str(line).strip(), e))
                if self.debugPrint:
                    sys.stderr.write(traceback.format_exc())
                sys.exit(1)
            lineNo = lineNo + 1

        if (self.state != 'none'):
            sys.stderr.write('Error when parsing file %s: File ended unexpectedly. Parser state: %s; Current Callee: %s; Current Subprocedure name: %s\n' \
                %(str(fileName), self.state, self.currCalleeName, self.currSubprocName))
            sys.exit(1)

class H90XMLCallGraphGenerator(H90CallGraphParser):
    doc = None
    routines = None
    calls = None
    currSubprocNode = None
    currDomainDependantRelationNode = None

    def __init__(self, doc):
        self.doc = doc
        self.routines = createOrGetFirstNodeWithName('routines', doc)
        self.calls = createOrGetFirstNodeWithName('calls', doc)
        super(H90XMLCallGraphGenerator, self).__init__()

    def processCallMatch(self, subProcCallMatch):
        super(H90XMLCallGraphGenerator, self).processCallMatch(subProcCallMatch)
        subProcName = subProcCallMatch.group(1)
        call = self.doc.createElement('call')
        call.setAttribute('caller', self.currSubprocName)
        call.setAttribute('callee', subProcName)
        if self.state == "inside_parallelRegion" or self.stateBeforeCall == "inside_parallelRegion":
            call.setAttribute('parallelRegionPosition', 'surround')
        if (not firstDuplicateChild(self.calls, call)):
            self.calls.appendChild(call)

    def processProcBeginMatch(self, subProcBeginMatch):
        super(H90XMLCallGraphGenerator, self).processProcBeginMatch(subProcBeginMatch)

        routine = self.doc.createElement('routine')
        routine.setAttribute('name', self.currSubprocName)
        self.routines.appendChild(routine)
        self.currSubprocNode = routine

    def processParallelRegionMatch(self, parallelRegionMatch):
        super(H90XMLCallGraphGenerator, self).processParallelRegionMatch(parallelRegionMatch)
        setTemplateInfos(self.doc, self.currSubprocNode, parallelRegionMatch.group(1), "parallelRegionTemplates", \
            "parallelRegionTemplate", "parallelRegions")

    def processDomainDependantMatch(self, domainDependantMatch):
        super(H90XMLCallGraphGenerator, self).processDomainDependantMatch(domainDependantMatch)
        self.currDomainDependantRelationNode = setTemplateInfos(self.doc, self.currSubprocNode, domainDependantMatch.group(1), "domainDependantTemplates", \
            "domainDependantTemplate", "domainDependants")

    def processDomainDependantEndMatch(self, domainDependantEndMatch):
        super(H90XMLCallGraphGenerator, self).processDomainDependantEndMatch(domainDependantEndMatch)
        self.currDomainDependantRelationNode = None

    def processProcEndMatch(self, subProcEndMatch):
        super(H90XMLCallGraphGenerator, self).processProcEndMatch(subProcEndMatch)
        self.currSubprocNode = None

    def processInsideDomainDependantRegionState(self, line):
        super(H90XMLCallGraphGenerator, self).processInsideDomainDependantRegionState(line)
        if self.state != 'inside_domainDependantRegion':
            return
        appendSeparatedTextAsNodes(line, ',', self.doc, self.currDomainDependantRelationNode, 'entry')


class H90CallGraphAndSymbolDeclarationsParser(H90CallGraphParser):
    cgDoc = None
    currSymbolsByName = {}
    symbolsOnCurrentLine = []
    importsOnCurrentLine = []
    routineNodesByProcName = {}
    parallelRegionTemplatesByProcName = {}

    def __init__(self, cgDoc):
        self.cgDoc = cgDoc
        self.currSymbolsByName = {}
        self.parallelRegionTemplatesByProcName = {}
        self.symbolsOnCurrentLine = []
        self.importsOnCurrentLine = []

        #build up dictionary of parallel regions by procedure name
        regionsByID = regionTemplatesByID(cgDoc, 'parallelRegionTemplate')
        routines = cgDoc.getElementsByTagName('routine')
        for routine in routines:
            procName = routine.getAttribute('name')
            if not procName or procName == '':
                raise Exception("Procedure without name.")
            self.routineNodesByProcName[procName] = routine

            parallelRegionsParents = routine.getElementsByTagName('activeParallelRegions')
            if not parallelRegionsParents or len(parallelRegionsParents) == 0:
                continue
            templateRelations = parallelRegionsParents[0].getElementsByTagName('templateRelation')
            regionTemplates = []
            for templateRelation in templateRelations:
                idStr = templateRelation.getAttribute('id')
                if not idStr or idStr == '':
                    raise Exception("Template relation without id attribute.")
                regionTemplate = regionsByID.get(idStr, None)
                if not regionTemplate:
                    raise Exception("Template relation id %s could not be matched in procedure '%s'" %(idStr, procName))
                regionTemplates.append(regionTemplate)

            if len(regionTemplates) > 0:
                self.parallelRegionTemplatesByProcName[procName] = regionTemplates

        super(H90CallGraphAndSymbolDeclarationsParser, self).__init__()

    def processSymbolDeclMatch(self, paramDeclMatch, symbol):
        '''process everything that happens per h90 declaration symbol'''
        symbol.isMatched = True
        symbol.loadDeclaration(paramDeclMatch, self.patterns)

    def processSymbolImportMatch(self, importMatch, symbol):
        symbol.isMatched = True
        symbol.loadImportInformation(importMatch)

    def processInsideDeclarationsState(self, line):
        '''process everything that happens per h90 declaration line'''
        super(H90CallGraphAndSymbolDeclarationsParser, self).processInsideDeclarationsState(line)
        if self.state != "inside_declarations":
            return

        branchMatch = self.patterns.branchPattern.match(str(line))
        if branchMatch:
            self.processBranchMatch(branchMatch)
            return

        symbolNames = self.currSymbolsByName.keys()
        for symbolName in symbolNames:
            symbol = self.currSymbolsByName[symbolName]
            declMatch = symbol.getDeclarationMatch(str(line))
            importMatch = symbol.symbolImportPattern.match(str(line))
            if declMatch:
                self.symbolsOnCurrentLine.append(symbol)
                self.processSymbolDeclMatch(declMatch, symbol)
            elif importMatch:
                self.importsOnCurrentLine.append(symbol)
                self.processSymbolImportMatch(importMatch, symbol)

        #validate the symbols on the current declaration line: Do they match the requirements for Hybrid Fortran?
        lineDeclarationType = DeclarationType.UNDEFINED
        for symbol in self.symbolsOnCurrentLine:
            if lineDeclarationType == DeclarationType.UNDEFINED:
                lineDeclarationType = symbol.declarationType()
            elif lineDeclarationType != symbol.declarationType():
                raise Exception("Symbols with different declaration types have been matched on the same line. This is invalid in Hybrid Fortran.\n" + \
                    "Example: Local arrays cannot be mixed with local scalars on the same declaration line. Please move apart these declarations.")

    def processProcBeginMatch(self, subProcBeginMatch):
        super(H90CallGraphAndSymbolDeclarationsParser, self).processProcBeginMatch(subProcBeginMatch)
        subprocName = subProcBeginMatch.group(1)
        routineNode = self.routineNodesByProcName.get(subprocName)
        if not routineNode:
            raise Exception("no definition found for routine '%s'" %(subprocName))
        #$$$ this might be dangerous: We're executing this at a time when parallel regions haven't been analyzed for CPU or GPU!
        #check whether this is really necessary and how it works.
        parallelRegionTemplates = self.parallelRegionTemplatesByProcName.get(self.currSubprocName)

        templatesAndEntries = getDomainDependantTemplatesAndEntries(self.cgDoc, routineNode)
        for template, entry in templatesAndEntries:
            dependantName = entry.firstChild.nodeValue
            symbol = Symbol(dependantName, template)
            symbol.loadDomainDependantEntryNodeAttributes(entry)
            symbol.loadRoutineNodeAttributes(routineNode, parallelRegionTemplates)
            self.currSymbolsByName[dependantName] = symbol

    def processProcEndMatch(self, subProcEndMatch):
        super(H90CallGraphAndSymbolDeclarationsParser, self).processProcEndMatch(subProcEndMatch)
        dependants = self.currSymbolsByName.keys()
        unmatched = []
        for dependant in dependants:
            if self.currSymbolsByName[dependant].isMatched:
                continue
            if len(self.currSymbolsByName[dependant].domains) == 0:
                #scalars that haven't been declared: Assume that they're from the local module
                self.currSymbolsByName[dependant].sourceModule = "HF90_LOCAL_MODULE"
                continue
            unmatched.append(dependant)
        if len(unmatched) != 0:
            raise Exception("The following non-scalar domain dependant declarations could not be found within subroutine %s: %s;\n\
                domains of first unmatched: %s"
                %(self.currSubprocName, unmatched, str(self.currSymbolsByName[unmatched[0]].domains))
            )
        self.currSymbolsByName = {}

    def processBranchMatch(self, branchMatch):
        branchSettingText = branchMatch.group(1).strip()
        branchSettings = branchSettingText.split(",")
        if len(branchSettings) != 1:
            raise Exception("Invalid number of branch settings.")
        branchSettingMatch = re.match(r'(\w*)\s*\(\s*(\w*)\s*\)', branchSettings[0].strip())
        if not branchSettingMatch:
            raise Exception("Invalid branch setting definition.")
        if branchSettingMatch.group(1) != "parallelRegion":
            raise Exception("Invalid branch setting definition: Currently only parallelRegion setting accepted.")
        self.stateBeforeBranch = self.state
        if branchSettingMatch.group(2) != self.routineNodesByProcName[self.currSubprocName].getAttribute('parallelRegionPosition').strip():
            self.state = 'inside_branch'
        else:
            self.state = 'inside_ignore'

    def processInsideBranch(self, line):
        if self.patterns.branchEndPattern.match(str(line)):
            self.state = self.stateBeforeBranch
            self.stateBeforeBranch = None

    def processInsideIgnore(self, line):
        if self.patterns.branchEndPattern.match(str(line)):
            self.state = self.stateBeforeBranch
            self.stateBeforeBranch = None

    def processLine(self, line):
        super(H90CallGraphAndSymbolDeclarationsParser, self).processLine(line)
        self.symbolsOnCurrentLine = []
        self.importsOnCurrentLine = []

class H90XMLSymbolDeclarationExtractor(H90CallGraphAndSymbolDeclarationsParser):

    entryNodesBySymbolName = {}
    currSymbols = []

    def processSymbolAttributes(self):
        currSymbolNames = self.currSymbolsByName.keys()
        if len(currSymbolNames) == 0:
            return
        self.currSymbols = [self.currSymbolsByName[symbolName] for symbolName in currSymbolNames]

        routineNode = self.routineNodesByProcName[self.currSubprocName]
        domainDependantsParentNodes = routineNode.getElementsByTagName("domainDependants")
        if domainDependantsParentNodes == None or len(domainDependantsParentNodes) == 0:
            raise Exception("Unexpected error: No domain dependant parent node found for routine %s where it has been identified before." %(self.currSubprocName))
        domainDependantsParentNode = domainDependantsParentNodes[0]
        domainDependantEntryNodes = domainDependantsParentNode.getElementsByTagName("entry")
        if domainDependantEntryNodes == None or len(domainDependantEntryNodes) == 0:
            raise Exception("Unexpected error: No domain dependants found for routine %s where they have been identified before." %(self.currSubprocName))
        self.entryNodesBySymbolName = {}
        for domainDependantEntryNode in domainDependantEntryNodes:
            self.entryNodesBySymbolName[domainDependantEntryNode.firstChild.nodeValue.strip()] = domainDependantEntryNode

        if len(self.entryNodesBySymbolName.keys()) != len(currSymbolNames):
            raise Exception("Unexpected error: %i domain dependant entry nodes found, when %i expected." \
                %(len(self.entryNodesBySymbolName.keys())), len(currSymbolNames))

    def processProcEndMatch(self, subProcEndMatch):
        #get handles to currently active symbols -> temporarily save the handles
        self.processSymbolAttributes()

        #finish parsing -> superclass destroys handles
        super(H90XMLSymbolDeclarationExtractor, self).processProcEndMatch(subProcEndMatch)

        #store our symbol informations to the xml
        for symbol in self.currSymbols:
            entryNode = self.entryNodesBySymbolName[symbol.name]
            if not entryNode:
                raise Exception("Unexpected error: symbol named %s not expected" %(symbol.name))
            symbol.storeDomainDependantEntryNodeAttributes(entryNode)

        #throw away our handles
        self.entryNodesBySymbolName = {}
        self.currSymbols = []


class H90toF90Printer(H90CallGraphAndSymbolDeclarationsParser):
    currSubroutineImplementationNeedsToBeCommented = False

    currRoutineIsCallingParallelRegion = False
    currCalleeNode = None
    additionalSymbolsByCalleeName = {}
    currAdditionalSubroutineParameters = []
    symbolsPassedInCurrentCallByName = {}

    currParallelIterators = []
    intentPattern = None
    dimensionPattern = None

    tab_insideSub = "\t\t"
    tab_outsideSub = "\t"

    implementation = None
    codeSanitizer = None

    stateBeforeBranch = None

    def __init__(self, cgDoc, implementation):
        self.implementation = implementation
        self.currRoutineIsCallingParallelRegion = False
        self.currSubroutineImplementationNeedsToBeCommented = False
        self.symbolsPassedInCurrentCallByName = {}
        self.additionalSymbolsByCalleeName = {}
        self.currParallelIterators = []
        self.currentLine = ""
        self.currCalleeNode = None
        self.currAdditionalSubroutineParameters = []
        self.codeSanitizer = FortranCodeSanitizer()

        super(H90toF90Printer, self).__init__(cgDoc)

    def prepareActiveParallelRegion(self, implementationFunctionName):
        routineNode = self.routineNodesByProcName.get(self.currSubprocName)
        if not routineNode:
            raise Exception("no definition found for routine '%s'", self.currSubprocName)
        if routineNode.getAttribute('parallelRegionPosition') != 'within':
            return
        templates = self.parallelRegionTemplatesByProcName.get(self.currSubprocName)
        if not templates or len(templates) == 0:
            raise Exception("Unexpected: no parallel template definition found for routine '%s'" \
                %(self.currSubprocName))
        if len(templates) > 1:
            raise Exception("Unexpected: more than one parallel region templates found for subroutine '%s' containing a parallelRegion directive" \
                %(self.currSubprocName))

        implementationAttr = getattr(self, 'implementation')
        functionAttr = getattr(implementationAttr, implementationFunctionName)
        self.prepareLine(functionAttr(templates[0]), self.tab_insideSub)

    def processSymbolMatchAndGetAdjustedLine(self, line, paramDeclMatch, symbol, isInsideSubroutineCall):
        #match the symbol's postfix again in the current given line. (The prefix could have changed from the last match.)
        postfix = paramDeclMatch.group(2)
        postfixEscaped = re.escape(postfix)
        pattern1 = r"(.*?)" + re.escape(symbol.deviceName()) + postfixEscaped + r"\s*"
        currMatch = re.match(pattern1, line)
        if not currMatch:
            pattern2 = r"(.*?)" + re.escape(symbol.name) + postfixEscaped + r"\s*"
            currMatch = re.match(pattern2, line)
            if not currMatch:
                raise Exception(\
                    "Symbol %s is accessed in an unexpected way. Note: '_d' postfix is reserved for internal use. Cannot match one of the following patterns: \npattern1: '%s'\npattern2: '%s'" \
                    %(symbol.name, pattern1, pattern2))


        prefix = currMatch.group(1)
        numOfIndependentDomains = len(symbol.domains) - symbol.numOfParallelDomains
        accPatternStr = None
        if numOfIndependentDomains > 0:
            accPatternStr = r"\s*\("
            for i in range(numOfIndependentDomains):
                if i != 0:
                    accPatternStr = accPatternStr + r"\,"
                accPatternStr = accPatternStr + r"\s*([\w\s\+\*\/\-\:]*)\s*"
            accPatternStr = accPatternStr + r"\)(.*)"
        elif numOfIndependentDomains == 0:
            bracketMatch = re.match(r"\s*\(.*", postfix)
            if bracketMatch:
                raise Exception("Unexpected array access for symbol %s: No match for access in domain %s with no independant dimensions. \
Note: Parallel domain accesses are inserted automatically." \
                    %(symbol.name, str(symbol.domains))
                )
        else:
            raise Exception("Number of parallel domains exceeds total number of domains with symbol %s" %(symbol.name))

        offsets = []
        if accPatternStr:
            accMatch = re.match(accPatternStr, postfix)
            if accMatch:
                #the number of groups should be numOfIndependentDomains plus 1 for the new postfix
                postfix = accMatch.group(numOfIndependentDomains + 1)
                for i in range(numOfIndependentDomains):
                    offsets.append(accMatch.group(i + 1))
            elif isInsideSubroutineCall:
                for i in range(numOfIndependentDomains):
                    offsets.append(":")
            else:
                raise Exception("Symbol %s is accessed in an unexpected way: Unexpected number of independent dimensions. \
access pattern: %s; access string: %s; domains: %s" \
                    %(symbol.name, accPatternStr, postfix, str(symbol.domains)))

        iterators = self.currParallelIterators
        if isInsideSubroutineCall:
            calleeNode = self.routineNodesByProcName.get(self.currCalleeName)
            if calleeNode and calleeNode.getAttribute("parallelRegionPosition") != "outside":
                iterators = []

        return (prefix + symbol.accessRepresentation(iterators, offsets) + postfix).rstrip() + "\n"

    def processSymbolsAndGetAdjustedLine(self, line, isInsideSubroutineCall):
        symbolNames = self.currSymbolsByName.keys()
        #TODO: in case of subroutine call, use the call routine domaindependant entries instead.
        adjustedLine = line
        for symbolName in symbolNames:
            symbol = self.currSymbolsByName[symbolName]
            symbolWasMatched = False
            lineSections = []
            work = adjustedLine
            nextMatch = symbol.namePattern.match(work)
            while nextMatch:
                if symbol.domains and len(symbol.domains) > 0 and not isInsideSubroutineCall and self.state != "inside_parallelRegion" \
                and self.routineNodesByProcName[self.currSubprocName].getAttribute("parallelRegionPosition") != "outside":
                    sys.stderr.write("WARNING: Dependant symbol %s accessed outside of a parallel region or subroutine call in subroutine %s\n" \
                    %(symbol.name, self.currSubprocName))

                symbolWasMatched = True
                prefix = nextMatch.group(1)
                lineSections.append(prefix)
                lineSections.append(symbol.deviceName())
                postfix = nextMatch.group(2)
                processed = self.processSymbolMatchAndGetAdjustedLine(work, nextMatch, symbol, isInsideSubroutineCall)
                adjustedMatch = symbol.namePattern.match(processed)
                if not adjustedMatch:
                    raise Exception("Unexpected error: symbol %s can't be matched again after adjustment. Adjusted portion: %s" %(symbol.name, processed))
                work = adjustedMatch.group(2)
                nextMatch = symbol.namePattern.match(work)
            #whatever is left now as "work" is the unmatched trailer of the line
            lineSections.append(work)
            adjustedLine = ""
            for section in lineSections:
                adjustedLine = adjustedLine + section
            if not isInsideSubroutineCall:
                continue
            if symbolWasMatched:
                self.symbolsPassedInCurrentCallByName[symbolName] = symbol
        return adjustedLine.rstrip() + "\n"

    def processSymbolImportAndGetAdjustedLine(self, importMatch):
        return self.implementation.adjustImportForDevice(\
            importMatch.group(0), \
            self.routineNodesByProcName[self.currSubprocName].getAttribute('parallelRegionPosition')
        )

    def processCallPostAndGetAdjustedLine(self, line):
        allSymbolsPassedByName = self.symbolsPassedInCurrentCallByName.copy()
        additionalSymbols = self.additionalSymbolsByCalleeName.get(self.currCalleeName)
        additionalDeclarations = []
        if additionalSymbols:
            additionalDeclarations = additionalSymbols[1]
        for symbol in additionalDeclarations:
            allSymbolsPassedByName[symbol.name] = symbol

        adjustedLine = line + "\n" + self.implementation.kernelCallPost(allSymbolsPassedByName, self.currCalleeNode)
        self.symbolsPassedInCurrentCallByName = {}
        self.currCalleeNode = None
        return adjustedLine

    def processCallMatch(self, subProcCallMatch):
        super(H90toF90Printer, self).processCallMatch(subProcCallMatch)

        adjustedLine = "call " + self.currCalleeName
        self.currCalleeNode = self.routineNodesByProcName.get(self.currCalleeName)

        parallelRegionPosition = None
        if self.currCalleeNode:
            parallelRegionPosition = self.currCalleeNode.getAttribute("parallelRegionPosition")
        if self.currCalleeNode and parallelRegionPosition == "within":
            parallelTemplates = self.parallelRegionTemplatesByProcName.get(self.currCalleeName)
            if not parallelTemplates or len(parallelTemplates) == 0:
                raise Exception("Unexpected: Subprocedure %s's parallelRegionPosition is defined as 'within', but no parallel region template could be found." \
                    %(self.currCalleeName))
            adjustedLine = self.implementation.kernelCallPreparation(parallelTemplates[0])
            adjustedLine = adjustedLine + "call " + self.currCalleeName + " " + self.implementation.kernelCallConfig()

        if self.currCalleeNode \
        and getRoutineNodeInitStage(self.currCalleeNode) == RoutineNodeInitStage.DIRECTIVES_WITHOUT_PARALLELREGION_POSITION:
            #special case. see also processBeginMatch.
            self.prepareLine("! " + subProcCallMatch.group(0), "")
            return

        arguments = subProcCallMatch.group(2)
        paramListMatch = self.patterns.subprocFirstLineParameterListPattern.match(arguments)
        if not paramListMatch and len(arguments.strip()) > 0:
            raise Exception("Subprocedure arguments without enclosing brackets. This is invalid in Hybrid Fortran")
        additionalSymbolsSeparated = self.additionalSymbolsByCalleeName.get(self.currCalleeName)
        if additionalSymbolsSeparated:
            additionalSymbols = sorted(additionalSymbolsSeparated[0] + additionalSymbolsSeparated[1])
        else:
            additionalSymbols = []
        if len(additionalSymbols) > 0:
            adjustedLine = adjustedLine + "( &\n"
        else:
            adjustedLine = adjustedLine + "("
        symbolNum = 0
        bridgeStr = ", & !additional parameter inserted by framework\n" + self.tab_insideSub + "& "
        for symbol in additionalSymbols:
            adjustedLine = adjustedLine + self.tab_insideSub + symbol.automaticName()
            if symbolNum < len(additionalSymbols) - 1 or paramListMatch:
                adjustedLine = adjustedLine + bridgeStr
            symbolNum = symbolNum + 1
        if paramListMatch:
            adjustedLine = adjustedLine + self.processSymbolsAndGetAdjustedLine(paramListMatch.group(2), isInsideSubroutineCall=True)
        else:
            adjustedLine = adjustedLine + ")\n"

        if self.currCalleeNode and self.currCalleeNode.getAttribute("parallelRegionPosition") == "within":
            if self.state != "inside_subroutine_call":
                adjustedLine = self.processCallPostAndGetAdjustedLine(adjustedLine)

        self.prepareLine(adjustedLine, self.tab_insideSub)

    def processAdditionalSubroutineParametersAndGetAdjustedLine(self):
        adjustedLine = str(self.currentLine)
        if len(self.currAdditionalSubroutineParameters) == 0:
            return adjustedLine

        paramListMatch = self.patterns.subprocFirstLineParameterListPattern.match(adjustedLine)
        if paramListMatch:
            adjustedLine = paramListMatch.group(1) + " &\n" + self.tab_outsideSub + "& "
            paramListStr = paramListMatch.group(2).strip()
        else:
            adjustedLine = str(self.currentLine) + "( &\n" + self.tab_outsideSub + "& "
            paramListStr = ")"
        #adjusted line now contains only prefix, including the opening bracket
        symbolNum = 0
        for symbol in self.currAdditionalSubroutineParameters:
            adjustedLine = adjustedLine + symbol.deviceName()
            if symbolNum != len(self.currAdditionalSubroutineParameters) - 1 or len(paramListStr) > 1:
                adjustedLine = adjustedLine + ","
            adjustedLine = adjustedLine + " & !additional symbol inserted by framework \n" + self.tab_outsideSub + "& "
            symbolNum = symbolNum + 1
        return adjustedLine + paramListStr

    def processBranchMatch(self, branchMatch):
        super(H90toF90Printer, self).processBranchMatch(branchMatch)
        self.prepareLine("","")

    def processProcBeginMatch(self, subProcBeginMatch):
        super(H90toF90Printer, self).processProcBeginMatch(subProcBeginMatch)

        subprocName = subProcBeginMatch.group(1)
        routineNode = self.routineNodesByProcName.get(subprocName)
        if not routineNode:
            raise Exception("no definition found for routine '%s'" %(subprocName))

        #build list of additional subroutine parameters
        #(parameters that the user didn't specify but that are necessary based on the features of the underlying technology
        # and the symbols declared by the user, such us temporary arrays and imported symbols)
        self.currAdditionalSubroutineParameters = self.implementation.extractListOfAdditionalSubroutineSymbols(routineNode, self.currSymbolsByName)
        adjustedLine = self.processAdditionalSubroutineParametersAndGetAdjustedLine()

        #print line
        self.prepareLine(self.implementation.subroutinePrefix(routineNode) + " " + adjustedLine, self.tab_outsideSub)

        #analyse whether this routine is calling other routines that have a parallel region within
        #+ analyse the additional symbols that come up there
        if not routineNode.getAttribute("parallelRegionPosition") == "inside":
            return
        callsLibraries = self.cgDoc.getElementsByTagName("calls")
        if not callsLibraries or len(callsLibraries) == 0:
            raise Exception("Unexpected Error: Caller library not found.")
        calls = callsLibraries[0].getElementsByTagName("call")
        for call in calls:
            if call.getAttribute("caller") != self.currSubprocName:
                continue
            calleeName = call.getAttribute("callee")
            callee = self.routineNodesByProcName.get(calleeName)
            if not callee:
                continue
            additionalSymbolsForCallee = self.implementation.getAdditionalSubroutineSymbols( \
                self.cgDoc, \
                callee, \
                self.parallelRegionTemplatesByProcName.get(calleeName) \
            )
            self.additionalSymbolsByCalleeName[calleeName] = additionalSymbolsForCallee
            if callee.getAttribute("parallelRegionPosition") != "within":
                continue
            self.currRoutineIsCallingParallelRegion = True

    def processProcEndMatch(self, subProcEndMatch):
        self.prepareLine(self.implementation.subroutineEnd(self.currSymbolsByName.values(), self.currRoutineIsCallingParallelRegion), self.tab_insideSub)
        self.currRoutineIsCallingParallelRegion = False
        self.prepareLine(self.currentLine + subProcEndMatch.group(0), self.tab_outsideSub)
        self.additionalSymbolsByCalleeName = {}
        self.currAdditionalSubroutineParameters = []
        self.currSubroutineImplementationNeedsToBeCommented = False
        super(H90toF90Printer, self).processProcEndMatch(subProcEndMatch)

    def processParallelRegionMatch(self, parallelRegionMatch):
        self.prepareLine("", "")
        super(H90toF90Printer, self).processParallelRegionMatch(parallelRegionMatch)
        self.prepareActiveParallelRegion('parallelRegionBegin')

    def processParallelRegionEndMatch(self, parallelRegionEndMatch):
        self.prepareLine("", "")
        super(H90toF90Printer, self).processParallelRegionEndMatch(parallelRegionEndMatch)
        self.prepareActiveParallelRegion('parallelRegionEnd')
        self.currParallelIterators = []

    def processDomainDependantMatch(self, domainDependantMatch):
        super(H90toF90Printer, self).processDomainDependantMatch(domainDependantMatch)
        self.prepareLine("", "")

    def processDomainDependantEndMatch(self, domainDependantEndMatch):
        super(H90toF90Printer, self).processDomainDependantEndMatch(domainDependantEndMatch)
        self.prepareLine("", "")

    def processNoMatch(self):
        super(H90toF90Printer, self).processNoMatch()
        self.prepareLine(str(self.currentLine), "")

    def processInsideDeclarationsState(self, line):
        '''process everything that happens per h90 declaration line'''
        super(H90toF90Printer, self).processInsideDeclarationsState(line)
        routineNode = self.routineNodesByProcName.get(self.currSubprocName)
        if routineNode \
        and getRoutineNodeInitStage(routineNode) == RoutineNodeInitStage.DIRECTIVES_WITHOUT_PARALLELREGION_POSITION:
            #during analysis we've found that this routine had a parallel region directive but now
            #there is no relation to a parallel region anymore in the callgraph.
            #This is a special case where the programmer most probably commented out the call to this subroutine
            #=> go into skipped state where the whole subroutine body is printed out commented (don't compile the body)
            self.currSubroutineImplementationNeedsToBeCommented = True
            if self.state == "inside_declarations":
                self.prepareLine(line, "")
            else:
                self.prepareLine("! " + line, "")
            return

        if self.state != "inside_declarations" and self.state != "none" and self.state != "inside_subroutine_call":
            additionalDeclarationsStr = ""
            if len(self.additionalSymbolsByCalleeName.keys()) > 0:
                additionalDeclarationsStr = "\n" + self.tab_insideSub + \
                 "! ****** additional symbols inserted by framework to emulate device support of language features\n"

            #########################################################################
            # additional symbols for called kernel                                  #
            #########################################################################
            for calleeName in self.additionalSymbolsByCalleeName.keys():
                _, additionalDeclarations = self.additionalSymbolsByCalleeName[calleeName]
                for symbol in additionalDeclarations:
                    declType = symbol.declarationType()
                    if declType != DeclarationType.IMPORTED_SCALAR and declType != DeclarationType.LOCAL_ARRAY:
                        continue
                    additionalDeclarationsStr = additionalDeclarationsStr + \
                        self.tab_insideSub + self.implementation.adjustDeclarationForDevice( \
                            self.tab_insideSub + \
                                symbol.getDeclarationLineForAutomaticSymbol().strip(), \
                            self.patterns, \
                            [symbol], \
                            self.currRoutineIsCallingParallelRegion, \
                            self.routineNodesByProcName[self.currSubprocName].getAttribute('parallelRegionPosition') \
                        ).rstrip() + "\n"
                    if self.debugPrint:
                        sys.stderr.write("...In subroutine %s: Symbol %s additionally declared and passed to %s\n" \
                            %(self.currSubprocName, symbol, calleeName) \
                        )

            #########################################################################
            # additional symbols for ourselves                                      #
            #########################################################################
            ourSymbolsToAdd = sorted(
                [symbol for symbol in self.currAdditionalSubroutineParameters
                    if symbol.sourceModule != None and symbol.sourceModule != ""
                ]
            )
            for symbol in ourSymbolsToAdd:
                additionalDeclarationsStr = additionalDeclarationsStr + \
                    self.tab_insideSub + self.implementation.adjustDeclarationForDevice( \
                        self.tab_insideSub + \
                            symbol.getDeclarationLineForAutomaticSymbol().strip(), \
                        self.patterns, \
                        [symbol], \
                        self.currRoutineIsCallingParallelRegion, \
                        self.routineNodesByProcName[self.currSubprocName].getAttribute('parallelRegionPosition') \
                    ).rstrip() + "\n"
                if self.debugPrint:
                    sys.stderr.write("...In subroutine %s: Symbol %s additionally declared\n" \
                        %(self.currSubprocName, symbol) \
                    )

            if len(self.additionalSymbolsByCalleeName.keys()) > 0:
                additionalDeclarationsStr = additionalDeclarationsStr + "! ****** end additional symbols\n\n"
            parallelTemplate = None
            parallelTemplates = self.parallelRegionTemplatesByProcName.get(self.currSubprocName)
            if parallelTemplates and len(parallelTemplates) > 0:
                parallelTemplate = parallelTemplates[0]
            self.prepareLine(additionalDeclarationsStr + \
                self.implementation.declarationEnd( \
                    self.currSymbolsByName.values(), \
                    self.currRoutineIsCallingParallelRegion, \
                    self.routineNodesByProcName[self.currSubprocName], \
                    parallelTemplate \
                ), \
                self.tab_insideSub \
            )
        if self.state != "inside_declarations":
            return

        adjustedLine = line
        for symbol in self.symbolsOnCurrentLine:
            match = symbol.getDeclarationMatch(str(adjustedLine))
            if not match:
                raise Exception("Unexpected error: Symbol %s not found on a line where it has already been identified before. Current string to search: %s" \
                    %(symbol, adjustedLine))
            adjustedLine = symbol.getAdjustedDeclarationLine(match, \
                self.parallelRegionTemplatesByProcName.get(self.currSubprocName), \
                self.patterns.dimensionPattern \
            )

        if adjustedLine != line:
            adjustedLine = purgeDimensionAndGetAdjustedLine(adjustedLine, self.patterns)
            adjustedLine = str(adjustedLine).rstrip() + "\n"

        if len(self.symbolsOnCurrentLine) > 0:
            adjustedLine = self.implementation.adjustDeclarationForDevice(adjustedLine, \
                self.patterns, \
                self.symbolsOnCurrentLine, \
                self.currRoutineIsCallingParallelRegion, \
                self.routineNodesByProcName[self.currSubprocName].getAttribute('parallelRegionPosition') \
            )

        for symbol in self.importsOnCurrentLine:
            match = symbol.symbolImportPattern.match(str(adjustedLine))
            if not match:
                continue
            adjustedLine = self.processSymbolImportAndGetAdjustedLine(match)

        self.prepareLine(adjustedLine, self.tab_insideSub)

    def processInsideSubroutineBodyState(self, line):
        '''process everything that happens per h90 subroutine body line'''
        branchMatch = self.patterns.branchPattern.match(str(line))
        if branchMatch:
            self.processBranchMatch(branchMatch)
            return

        if self.patterns.branchEndPattern.match(str(line)):
            self.prepareLine("","")
            return

        subProcCallMatch = self.patterns.subprocCallPattern.match(str(line))
        if subProcCallMatch:
            self.processCallMatch(subProcCallMatch)
            return

        subProcEndMatch = self.patterns.subprocEndPattern.match(str(line))
        if (subProcEndMatch):
            self.processProcEndMatch(subProcEndMatch)
            self.state = 'none'
            self.currSubprocName = None
            return

        if self.currSubroutineImplementationNeedsToBeCommented:
            self.prepareLine("! " + line, "")
            return

        parallelRegionMatch = self.patterns.parallelRegionPattern.match(str(line))
        if (parallelRegionMatch):
            templates = self.parallelRegionTemplatesByProcName.get(self.currSubprocName)
            if not templates or len(templates) == 0:
                sys.stderr.write("WARNING: Routine %s contains an @parallelRegion statement but no parallel region information \
was found in the current callgraph.\n" %(self.currSubprocName))
            elif len(templates) != 1:
                raise Exception("Multiple parallel region templates found for subroutine %s. It is not allowed to have multiple \
parallel regions defined in the same subroutine in Hybrid Fortran." %(self.currSubprocName))
            elif self.routineNodesByProcName[self.currSubprocName].getAttribute('parallelRegionPosition') == "within":
                #to reach this block we have exactly one parallel region template
                self.currParallelIterators = self.implementation.getIterators(templates[0])
                if len(self.currParallelIterators) > 0:
                    self.processParallelRegionMatch(parallelRegionMatch)
                    self.state = 'inside_parallelRegion'
            if self.state != 'inside_parallelRegion':
                self.prepareLine("","")
            return

        if (self.patterns.parallelRegionEndPattern.match(str(line))):
            #note: this may occur when a parallel region is discarded because it doesn't apply
            #-> state stays within body and the region end line will trap here
            self.prepareLine("","")
            return

        domainDependantMatch = self.patterns.domainDependantPattern.match(str(line))
        if (domainDependantMatch):
            self.processDomainDependantMatch(domainDependantMatch)
            self.state = 'inside_domainDependantRegion'
            return

        if (self.patterns.subprocBeginPattern.match(str(line))):
            raise Exception("subprocedure within subprocedure not allowed")

        adjustedLine = self.processSymbolsAndGetAdjustedLine(line, False)
        self.prepareLine(adjustedLine, self.tab_insideSub)

    def processInsideParallelRegionState(self, line):
        branchMatch = self.patterns.branchPattern.match(str(line))
        if branchMatch:
            self.processBranchMatch(branchMatch)
            return

        subProcCallMatch = self.patterns.subprocCallPattern.match(str(line))
        if subProcCallMatch:
            if subProcCallMatch.group(1) not in self.routineNodesByProcName.keys():
                sys.stderr.write(self.implementation.warningOnUnrecognizedSubroutineCallInParallelRegion( \
                    self.currSubprocName, subProcCallMatch.group(1)))
            self.processCallMatch(subProcCallMatch)
            return

        parallelRegionEndMatch = self.patterns.parallelRegionEndPattern.match(str(line))
        if (parallelRegionEndMatch):
            self.processParallelRegionEndMatch(parallelRegionEndMatch)
            self.state = "inside_subroutine_body"
            return

        if (self.patterns.parallelRegionPattern.match(str(line))):
            raise Exception("parallelRegion within parallelRegion not allowed")
        if (self.patterns.subprocEndPattern.match(str(line))):
            raise Exception("subprocedure end before @end parallelRegion")
        if (self.patterns.subprocBeginPattern.match(str(line))):
            raise Exception("subprocedure within subprocedure not allowed")

        adjustedLine = self.processSymbolsAndGetAdjustedLine(line, isInsideSubroutineCall=False)
        self.prepareLine(adjustedLine, self.tab_insideSub)

    def processInsideSubroutineCall(self, line):
        super(H90toF90Printer, self).processInsideSubroutineCall(line)
        adjustedLine = ""

        if self.currCalleeNode and getRoutineNodeInitStage(self.currCalleeNode) == RoutineNodeInitStage.DIRECTIVES_WITHOUT_PARALLELREGION_POSITION:
            #special case. see also processBeginMatch.
            adjustedLine = "! " + line
        else:
            adjustedLine = self.processSymbolsAndGetAdjustedLine(line, isInsideSubroutineCall=True)

        if self.currCalleeNode and self.state != "inside_subroutine_call":
            adjustedLine = self.processCallPostAndGetAdjustedLine(adjustedLine)
        self.prepareLine(adjustedLine.rstrip() + "\n", "")

    def processInsideDomainDependantRegionState(self, line):
        super(H90toF90Printer, self).processInsideDomainDependantRegionState(line)
        self.prepareLine("", "")

    def processSpecificationBeginning(self):
        adjustedLine = self.currentLine
        for calleeName in self.additionalSymbolsByCalleeName.keys():
            additionalImports, _ = self.additionalSymbolsByCalleeName[calleeName]
            for symbol in additionalImports:
                declType = symbol.declarationType()
                if declType != DeclarationType.IMPORTED_SCALAR and declType != DeclarationType.LOCAL_ARRAY:
                    continue
                adjustedLine = adjustedLine + "use %s, only : %s => %s\n" \
                    %(symbol.sourceModule, symbol.automaticName(), symbol.sourceSymbol)

        self.prepareLine(adjustedLine + self.implementation.additionalIncludes(), self.tab_insideSub)

    def processInsideBranch(self, line):
        super(H90toF90Printer, self).processInsideBranch(line)
        self.prepareLine("", "")

    def processInsideIgnore(self, line):
        super(H90toF90Printer, self).processInsideIgnore(line)
        if self.state != "inside_ignore":
            self.prepareLine("", "")
            return
        self.prepareLine(line, "")

    def processLine(self, line):
        super(H90toF90Printer, self).processLine(line)
        sys.stdout.write(self.currentLine)

    def processFile(self, fileName):
        sys.stdout.write('''#include "storage_order.F90"\n''')
        super(H90toF90Printer, self).processFile(fileName)

    def prepareLine(self, line, tab):
        self.currentLine = self.codeSanitizer.sanitizeLines(line)
