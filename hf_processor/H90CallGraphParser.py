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
        self.openMPLinePattern = re.compile(r'\s*\!\$OMP.*', re.IGNORECASE)
        self.openACCLinePattern = re.compile(r'\s*\!\$acc.*', re.IGNORECASE)
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
            oldLineLength = -1
            blankPos = -1
            while len(currLine) > howManyCharsPerLine - len(lineSep):
                isOpenMPDirectiveLine = self.openMPLinePattern.match(currLine) != None
                isOpenACCDirectiveLine = self.openACCLinePattern.match(currLine) != None
                if commentChar in currLine and not isOpenMPDirectiveLine and not isOpenACCDirectiveLine:
                    commentPos = currLine.find(commentChar)
                    if commentPos <= howManyCharsPerLine:
                        break
                #find a blank that's NOT within a quoted string
                blankPos = findRightMostOccurrenceNotInsideQuotes(' ', currLine, rightStartAt=howManyCharsPerLine - len(lineSep))
                if blankPos < 1 or (oldLineLength != -1 and len(currLine[:blankPos]) >= oldLineLength):
                    #blank not found or at beginning of line
                    #-> bail out in order to avoid infinite loop - just keep the line as it was.
                    sys.stderr.write(
                        "WARNING: The following line could not be broken up for Fortran compatibility - no suitable spaces found: %s\n" %(
                            currLine,
                        )
                    )
                    sanitizedCodeLines.append(currLine)
                    currLine = ""
                    break
                else:
                    sanitizedCodeLines.append(currLine[:blankPos] + lineSep)
                    oldLineLength = len(currLine)
                    if isOpenMPDirectiveLine:
                        currLine = '!$OMP& ' + currLine[blankPos:]
                    elif isOpenACCDirectiveLine:
                        currLine = '!$acc& ' + currLine[blankPos:]
                    else:
                        currLine = '& ' + currLine[blankPos:]
            if toBeCommented:
                currLine = commentChar + " " + currLine
            if currLine != "":
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
                if self.currNumOfTabs < 5:
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
    currModuleName = None
    currBracketAnalyzer = None
    currCalleeName = None
    patterns = None
    currentLine = None
    branchAnalyzer = None
    currTemplateName = None

    def __init__(self):
        self.patterns = H90RegExPatterns()
        self.state = "none"
        self.stateBeforeCall = "undefined"
        self.currCalleeName = None
        self.branchAnalyzer = BracketAnalyzer(
            r'^\s*if\s*\(|^\s*select\s+case',
            r'^\s*end\s+if|^\s*end\s+select',
            pass_in_regex_pattern=True
        )
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

    def processModuleBeginMatch(self, moduleBeginMatch):
        return

    def processModuleEndMatch(self, moduleEndMatch):
        return

    def processTemplateMatch(self, templateMatch):
        settingPattern = re.compile(r'[\s,]*(\w*)\s*(\(.*)')
        settingMatch = settingPattern.match(templateMatch.group(1))
        if not settingMatch:
            self.currTemplateName = None
            return
        settingName = settingMatch.group(1).strip()
        if settingName != 'name':
            self.currTemplateName = None
            return
        textAfterSettingName = settingMatch.group(2)
        settingBracketAnalyzer = BracketAnalyzer()
        settingText, remainder = settingBracketAnalyzer.splitAfterClosingBrackets(textAfterSettingName)
        #cut away the left and right bracket
        settingText = settingText.partition("(")[2]
        settingText = settingText.rpartition(")")[0]
        self.currTemplateName = settingText

    def processTemplateEndMatch(self, templateEndMatch):
        self.currTemplateName = None

    def processNoMatch(self):
        return

    def processSpecificationBeginning(self):
        return

    def processNoneState(self, line):
        specificationsBeginHere = False
        moduleBeginMatch = self.patterns.moduleBeginPattern.match(str(line))
        subProcBeginMatch = self.patterns.subprocBeginPattern.match(str(line))
        templateMatch = self.patterns.templatePattern.match(str(line))
        templateEndMatch = self.patterns.templateEndPattern.match(str(line))
        if templateMatch:
            self.processTemplateMatch(templateMatch)
        elif templateEndMatch:
            self.processTemplateEndMatch(templateEndMatch)
        elif moduleBeginMatch:
            self.currModuleName = moduleBeginMatch.group(1)
            self.state = 'inside_module'
            self.processModuleBeginMatch(moduleBeginMatch)
        elif subProcBeginMatch:
            raise Exception("please put this Hybrid Fortran subroutine into a module")
        else:
            self.processNoMatch()

        if specificationsBeginHere:
            self.processSpecificationBeginning()

    def processInsideBranch(self, line):
        return

    def processInsideIgnore(self, line):
        return

    def processInsideModuleState(self, line):
        specificationsBeginHere = False
        subProcBeginMatch = self.patterns.subprocBeginPattern.match(str(line))
        moduleEndMatch = self.patterns.moduleEndPattern.match(str(line))
        domainDependantMatch = self.patterns.domainDependantPattern.match(str(line))
        templateMatch = self.patterns.templatePattern.match(str(line))
        templateEndMatch = self.patterns.templateEndPattern.match(str(line))

        if templateMatch:
            self.processTemplateMatch(templateMatch)

        elif templateEndMatch:
            self.processTemplateEndMatch(templateEndMatch)

        elif self.currBracketAnalyzer:
            level = self.currBracketAnalyzer.currLevelAfterString(line)
            if level == 0:
                self.state ='inside_declarations'
                self.currBracketAnalyzer = None
                specificationsBeginHere = True
            self.processNoMatch()

        elif domainDependantMatch:
            self.state = 'inside_moduleDomainDependantRegion'
            self.processDomainDependantMatch(domainDependantMatch)

        elif moduleEndMatch:
            self.processModuleEndMatch(moduleEndMatch)
            self.currModuleName = None
            self.state = 'none'

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
        templateMatch = self.patterns.templatePattern.match(str(line))
        templateEndMatch = self.patterns.templateEndPattern.match(str(line))

        if (domainDependantMatch):
            self.state = 'inside_domainDependantRegion'
            self.processDomainDependantMatch(domainDependantMatch)
        elif subProcCallMatch:
            self.processCallMatch(subProcCallMatch)
        elif (subProcEndMatch):
            self.processProcEndMatch(subProcEndMatch)
            self.state = 'inside_module'
            self.currBracketAnalyzer = None
            self.currSubprocName = None
        elif (parallelRegionMatch):
            raise Exception("parallel region without parallel dependants")
        elif (self.patterns.subprocBeginPattern.match(str(line))):
            raise Exception("subprocedure within subprocedure not allowed")
        elif templateMatch:
            raise Exception("template directives are only allowed outside of subroutines")
        elif templateEndMatch:
            raise Exception("template directives are only allowed outside of subroutines")
        return

    def processInsideSubroutineBodyState(self, line):
        #note: Branches (@if statements) are ignored here, we want to keep analyzing their statements for callgraphs.
        domainDependantMatch = self.patterns.domainDependantPattern.match(str(line))
        subProcCallMatch = self.patterns.subprocCallPattern.match(str(line))
        parallelRegionMatch = self.patterns.parallelRegionPattern.match(str(line))
        domainDependantMatch = self.patterns.domainDependantPattern.match(str(line))
        subProcEndMatch = self.patterns.subprocEndPattern.match(str(line))
        templateMatch = self.patterns.templatePattern.match(str(line))
        templateEndMatch = self.patterns.templateEndPattern.match(str(line))

        if domainDependantMatch:
            self.state = 'inside_domainDependantRegion'
            self.processDomainDependantMatch(domainDependantMatch)
        elif subProcCallMatch:
            self.processCallMatch(subProcCallMatch)
        elif subProcEndMatch:
            self.processProcEndMatch(subProcEndMatch)
            self.state = 'inside_module'
            self.currBracketAnalyzer = None
            self.currSubprocName = None
        elif parallelRegionMatch:
            self.processParallelRegionMatch(parallelRegionMatch)
            self.state = 'inside_parallelRegion'
        elif self.patterns.subprocBeginPattern.match(str(line)):
            raise Exception("subprocedure within subprocedure not allowed")
        elif templateMatch:
            raise Exception("template directives are only allowed outside of subroutines")
        elif templateEndMatch:
            raise Exception("template directives are only allowed outside of subroutines")
        return

    def processInsideParallelRegionState(self, line):
        subProcCallMatch = self.patterns.subprocCallPattern.match(str(line))
        parallelRegionEndMatch = self.patterns.parallelRegionEndPattern.match(str(line))
        templateMatch = self.patterns.templatePattern.match(str(line))
        templateEndMatch = self.patterns.templateEndPattern.match(str(line))
        if subProcCallMatch:
            self.processCallMatch(subProcCallMatch)
        elif (parallelRegionEndMatch):
            self.processParallelRegionEndMatch(parallelRegionEndMatch)
            self.state = "inside_subroutine_body"
        elif (self.patterns.earlyReturnPattern.match(str(line))):
            raise Exception("early return in the same subroutine within parallelRegion not allowed")
        elif (self.patterns.parallelRegionPattern.match(str(line))):
            raise Exception("parallelRegion within parallelRegion not allowed")
        elif (self.patterns.subprocEndPattern.match(str(line))):
            raise Exception("subprocedure end before @end parallelRegion")
        elif (self.patterns.subprocBeginPattern.match(str(line))):
            raise Exception("subprocedure within subprocedure not allowed")
        elif templateMatch:
            raise Exception("template directives are only allowed outside of subroutines")
        elif templateEndMatch:
            raise Exception("template directives are only allowed outside of subroutines")
        else:
            self.processNoMatch()

    def processInsideModuleDomainDependantRegionState(self, line):
        domainDependantEndMatch = self.patterns.domainDependantEndPattern.match(str(line))
        templateMatch = self.patterns.templatePattern.match(str(line))
        templateEndMatch = self.patterns.templateEndPattern.match(str(line))
        if domainDependantEndMatch:
            self.processDomainDependantEndMatch(domainDependantEndMatch)
            self.state = "inside_module"
        elif (self.patterns.earlyReturnPattern.match(str(line))):
            raise Exception("early return not allowed here")
        elif self.patterns.subprocCallPattern.match(str(line)):
            raise Exception("subprocedure call within domainDependants not allowed")
        elif (self.patterns.parallelRegionEndPattern.match(str(line)) or self.patterns.parallelRegionPattern.match(str(line))):
            raise Exception("parallelRegion within domainDependants not allowed")
        elif (self.patterns.subprocEndPattern.match(str(line))):
            raise Exception("subprocedure end before @end domainDependant")
        elif (self.patterns.subprocBeginPattern.match(str(line))):
            raise Exception("subprocedure within subprocedure not allowed")
        elif templateMatch:
            raise Exception("template directives not allowed here")
        elif templateEndMatch:
            raise Exception("template directives not allowed here")
        return

    def processInsideDomainDependantRegionState(self, line):
        domainDependantEndMatch = self.patterns.domainDependantEndPattern.match(str(line))
        templateMatch = self.patterns.templatePattern.match(str(line))
        templateEndMatch = self.patterns.templateEndPattern.match(str(line))
        if domainDependantEndMatch:
            self.processDomainDependantEndMatch(domainDependantEndMatch)
            self.state = "inside_subroutine_body"
        elif (self.patterns.earlyReturnPattern.match(str(line))):
            raise Exception("early return not allowed here")
        elif self.patterns.subprocCallPattern.match(str(line)):
            raise Exception("subprocedure call within domainDependants not allowed")
        elif (self.patterns.parallelRegionEndPattern.match(str(line)) or self.patterns.parallelRegionPattern.match(str(line))):
            raise Exception("parallelRegion within domainDependants not allowed")
        elif (self.patterns.subprocEndPattern.match(str(line))):
            raise Exception("subprocedure end before @end domainDependant")
        elif (self.patterns.subprocBeginPattern.match(str(line))):
            raise Exception("subprocedure within subprocedure not allowed")
        elif templateMatch:
            raise Exception("template directives not allowed here")
        elif templateEndMatch:
            raise Exception("template directives not allowed here")
        return

    def processUndefinedState(self, line):
        raise Exception("unexpected undefined parser state: %s" %(self.state))

    def processLine(self, line):
        #define the states and their respective handlers
        stateSwitch = {
           'none': self.processNoneState,
           'inside_module': self.processInsideModuleState,
           'inside_moduleDomainDependantRegion': self.processInsideModuleDomainDependantRegionState,
           'inside_declarations': self.processInsideDeclarationsState,
           'inside_parallelRegion': self.processInsideParallelRegionState,
           'inside_domainDependantRegion': self.processInsideDomainDependantRegionState,
           'inside_subroutine_body': self.processInsideSubroutineBodyState,
           'inside_subroutine_call': self.processInsideSubroutineCall,
           'inside_branch': self.processInsideBranch,
           'inside_ignore': self.processInsideIgnore
         }
        #exclude commented lines from analysis
        if (self.patterns.commentedPattern.match(str(line))):
            self.currentLine = line
            return

        #remove trailing comments
        self.currentLine = self.purgeTrailingCommentsAndGetAdjustedLine(str(line))

        self.branchAnalyzer.currLevelAfterString(str(line))

        #analyse this line. handle the line according to current parser state.
        stateSwitch.get(self.state, self.processUndefinedState)(self.currentLine)
        if not self.state == "inside_subroutine_call":
            self.currCalleeName = None

    def processFile(self, fileName):
        self.lineNo = 1
        self.fileName = fileName
        for line in fileinput.input([fileName]):
            try:
                self.processLine(line)
            except Exception, e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                sys.stderr.write('Error when parsing file %s on line %i: %s; Print of line:\n%s\n' %(
                        str(fileName), self.lineNo, str(e), str(line).strip()
                    )
                )
                if self.debugPrint:
                    sys.stderr.write(traceback.format_exc())
                sys.exit(1)
            self.lineNo += 1

        if (self.state != 'none'):
            sys.stderr.write('Error when parsing file %s: File ended unexpectedly. Parser state: %s; Current Callee: %s; Current Subprocedure name: %s\n' \
                %(str(fileName), self.state, self.currCalleeName, self.currSubprocName))
            sys.exit(1)
        del self.lineNo
        del self.fileName

class H90XMLCallGraphGenerator(H90CallGraphParser):
    doc = None
    routines = None
    modules = None
    templates = None
    calls = None
    currSubprocNode = None
    currModuleNode = None
    currDomainDependantRelationNode = None
    currParallelRegionTemplateNode = None
    currParallelRegionRelationNode = None

    def __init__(self, doc):
        self.doc = doc
        self.routines = createOrGetFirstNodeWithName('routines', doc)
        self.calls = createOrGetFirstNodeWithName('calls', doc)
        self.modules = createOrGetFirstNodeWithName('modules', doc)
        self.templates = createOrGetFirstNodeWithName('implementationTemplates', doc)
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

    def processTemplateMatch(self, templateMatch):
        super(H90XMLCallGraphGenerator, self).processTemplateMatch(templateMatch)
        template = self.doc.createElement('implementationTemplate')
        template.setAttribute('name', self.currTemplateName)
        self.templates.appendChild(template)
        return

    def processModuleBeginMatch(self, moduleBeginMatch):
        super(H90XMLCallGraphGenerator, self).processModuleBeginMatch(moduleBeginMatch)
        module = self.doc.createElement('module')
        module.setAttribute('name', self.currModuleName)
        self.modules.appendChild(module)
        self.currModuleNode = module

    def processProcBeginMatch(self, subProcBeginMatch):
        super(H90XMLCallGraphGenerator, self).processProcBeginMatch(subProcBeginMatch)
        routine = self.doc.createElement('routine')
        routine.setAttribute('name', self.currSubprocName)
        routine.setAttribute('source', os.path.basename(self.fileName).split('.')[0])
        self.routines.appendChild(routine)
        self.currSubprocNode = routine

    def processParallelRegionMatch(self, parallelRegionMatch):
        super(H90XMLCallGraphGenerator, self).processParallelRegionMatch(parallelRegionMatch)
        self.currParallelRegionRelationNode, self.currParallelRegionTemplateNode = setTemplateInfos(
            self.doc,
            self.currSubprocNode,
            parallelRegionMatch.group(1),
            "parallelRegionTemplates",
            "parallelRegionTemplate",
            "parallelRegions"
        )
        self.currParallelRegionRelationNode.setAttribute("startLine", str(self.lineNo))

    def processParallelRegionEndMatch(self, parallelRegionEndMatch):
        super(H90XMLCallGraphGenerator, self).processParallelRegionEndMatch(parallelRegionEndMatch)
        self.currParallelRegionRelationNode.setAttribute("endLine", str(self.lineNo))
        self.currParallelRegionTemplateNode = None
        self.currParallelRegionRelationNode = None

    def processDomainDependantMatch(self, domainDependantMatch):
        super(H90XMLCallGraphGenerator, self).processDomainDependantMatch(domainDependantMatch)
        self.currDomainDependantRelationNode, _ = setTemplateInfos(
            self.doc,
            self.currModuleNode if self.state == 'inside_moduleDomainDependantRegion' else self.currSubprocNode,
            domainDependantMatch.group(1),
            "domainDependantTemplates",
            "domainDependantTemplate",
            "domainDependants"
        )

    def processDomainDependantEndMatch(self, domainDependantEndMatch):
        super(H90XMLCallGraphGenerator, self).processDomainDependantEndMatch(domainDependantEndMatch)
        self.currDomainDependantRelationNode = None

    def processProcEndMatch(self, subProcEndMatch):
        super(H90XMLCallGraphGenerator, self).processProcEndMatch(subProcEndMatch)
        self.currSubprocNode = None

    def processModuleEndMatch(self, moduleEndMatch):
        super(H90XMLCallGraphGenerator, self).processModuleEndMatch(moduleEndMatch)
        self.currModuleNode = None

    def processInsideModuleDomainDependantRegionState(self, line):
        super(H90XMLCallGraphGenerator, self).processInsideModuleDomainDependantRegionState(line)
        if self.state != 'inside_moduleDomainDependantRegion':
            return
        appendSeparatedTextAsNodes(line, ',', self.doc, self.currDomainDependantRelationNode, 'entry')

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
    moduleNodesByName = {}
    parallelRegionTemplatesByProcName = {}
    parallelRegionTemplateRelationsByProcName = {}

    def __init__(self, cgDoc):
        self.cgDoc = cgDoc
        self.currSymbolsByName = {}
        self.parallelRegionTemplatesByProcName = {}
        self.parallelRegionTemplateRelationsByProcName = {}
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
            if len(templateRelations) > 0:
                self.parallelRegionTemplateRelationsByProcName[procName] = templateRelations
            if len(regionTemplates) > 0:
                self.parallelRegionTemplatesByProcName[procName] = regionTemplates
        modules = cgDoc.getElementsByTagName('module')
        for module in modules:
            moduleName = module.getAttribute('name')
            if not moduleName or moduleName == '':
                raise Exception("Module without name.")
            self.moduleNodesByName[moduleName] = module

        super(H90CallGraphAndSymbolDeclarationsParser, self).__init__()

    def loadSymbolsFromTemplate(self, parentNode, parallelRegionTemplates, isModuleSymbols=False):
        templatesAndEntries = getDomainDependantTemplatesAndEntries(self.cgDoc, parentNode)
        for template, entry in templatesAndEntries:
            dependantName = entry.firstChild.nodeValue
            symbol = Symbol(dependantName, template, debugPrint=self.debugPrint)
            symbol.isModuleSymbol = isModuleSymbols
            symbol.loadDomainDependantEntryNodeAttributes(entry)
            if isModuleSymbols:
                symbol.loadModuleNodeAttributes(parentNode)
            else:
                symbol.loadRoutineNodeAttributes(parentNode, parallelRegionTemplates)
            self.currSymbolsByName[dependantName] = symbol
        if self.debugPrint:
            sys.stderr.write("Symbols loaded from template. Symbols currently active in scope: %s. Module Symbol Property: %s\n" %(
                str(self.currSymbolsByName.values()),
                str([self.currSymbolsByName[symbolName].isModuleSymbol for symbolName in self.currSymbolsByName.keys()])
            ))

    def analyseSymbolInformationOnCurrentLine(self, line, analyseImports=True):
        symbolNames = self.currSymbolsByName.keys()
        for symbolName in symbolNames:
            symbol = self.currSymbolsByName[symbolName]
            declMatch = symbol.getDeclarationMatch(str(line))
            importMatch = None
            if analyseImports:
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

    def processSymbolDeclMatch(self, paramDeclMatch, symbol):
        '''process everything that happens per h90 declaration symbol'''
        symbol.isMatched = True
        symbol.loadDeclaration(paramDeclMatch, self.patterns)

    def processSymbolImportMatch(self, importMatch, symbol):
        symbol.isMatched = True
        moduleName = importMatch.group(1)
        moduleNode = self.moduleNodesByName.get(moduleName)
        symbol.loadImportInformation(importMatch, self.cgDoc, moduleNode)

    def processInsideModuleState(self, line):
        super(H90CallGraphAndSymbolDeclarationsParser, self).processInsideModuleState(line)
        self.analyseSymbolInformationOnCurrentLine(line, analyseImports=False)

    def processInsideDeclarationsState(self, line):
        '''process everything that happens per h90 declaration line'''
        super(H90CallGraphAndSymbolDeclarationsParser, self).processInsideDeclarationsState(line)
        if self.state != "inside_declarations":
            return
        branchMatch = self.patterns.branchPattern.match(str(line))
        if branchMatch:
            self.processBranchMatch(branchMatch)
            return
        self.analyseSymbolInformationOnCurrentLine(line)

    def processModuleBeginMatch(self, moduleBeginMatch):
        super(H90CallGraphAndSymbolDeclarationsParser, self).processModuleBeginMatch(moduleBeginMatch)
        moduleName = moduleBeginMatch.group(1)
        moduleNode = self.moduleNodesByName.get(moduleName)
        if not moduleNode:
            return
        self.loadSymbolsFromTemplate(moduleNode, None, isModuleSymbols=True)

    def processModuleEndMatch(self, moduleEndMatch):
        super(H90CallGraphAndSymbolDeclarationsParser, self).processProcEndMatch(moduleEndMatch)
        dependants = self.currSymbolsByName.keys()
        unmatched = []
        for dependant in dependants:
            if not self.currSymbolsByName[dependant].isModuleSymbol:
                raise Exception("Dependant %s has been referenced in a domain dependant region inside a procedure, but has never been matched." %(dependant))
            if self.currSymbolsByName[dependant].isMatched:
                continue
            unmatched.append(dependant)
        if len(unmatched) != 0:
            raise Exception("The following non-scalar domain dependant declarations could not be found within module %s: %s;\n\
                domains of first unmatched: %s"
                %(self.currModuleName, unmatched, str(self.currSymbolsByName[unmatched[0]].domains))
            )
        if self.debugPrint:
            sys.stderr.write("Clearing current symbol scope since the module definition is finished\n")
        self.currSymbolsByName = {}

    def processProcBeginMatch(self, subProcBeginMatch):
        super(H90CallGraphAndSymbolDeclarationsParser, self).processProcBeginMatch(subProcBeginMatch)
        subprocName = subProcBeginMatch.group(1)
        routineNode = self.routineNodesByProcName.get(subprocName)
        if not routineNode:
            return
        parallelRegionTemplates = self.parallelRegionTemplatesByProcName.get(self.currSubprocName)
        self.loadSymbolsFromTemplate(routineNode, parallelRegionTemplates)

    def processProcEndMatch(self, subProcEndMatch):
        super(H90CallGraphAndSymbolDeclarationsParser, self).processProcEndMatch(subProcEndMatch)
        dependants = self.currSymbolsByName.keys()
        unmatched = []
        for dependant in dependants:
            if self.currSymbolsByName[dependant].isModuleSymbol:
                continue
            if self.currSymbolsByName[dependant].isMatched:
                del self.currSymbolsByName[dependant]
                continue
            if len(self.currSymbolsByName[dependant].domains) == 0:
                #scalars that haven't been declared: Assume that they're from the local module
                self.currSymbolsByName[dependant].sourceModule = "HF90_LOCAL_MODULE"
                self.currSymbolsByName[dependant].isModuleSymbol = True
                del self.currSymbolsByName[dependant]
                continue
            unmatched.append(dependant)
        if len(unmatched) != 0:
            raise Exception("The following non-scalar domain dependant declarations could not be found within subroutine %s: %s;\n\
                domains of first unmatched: %s"
                %(self.currSubprocName, unmatched, str(self.currSymbolsByName[unmatched[0]].domains))
            )

    def processBranchMatch(self, branchMatch):
        branchSettingText = branchMatch.group(1).strip()
        branchSettings = branchSettingText.split(",")
        if len(branchSettings) != 1:
            raise Exception("Invalid number of branch settings.")
        branchSettingMatch = re.match(r'(\w*)\s*\(\s*(\w*)\s*\)', branchSettings[0].strip(), re.IGNORECASE)
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

    def processSymbolAttributes(self, isModule=False):
        currSymbolNames = self.currSymbolsByName.keys()
        self.currSymbols = []
        if len(currSymbolNames) == 0:
            return
        self.currSymbols = [self.currSymbolsByName[symbolName] for symbolName in currSymbolNames if self.currSymbolsByName[symbolName].isModuleSymbol == isModule]
        if len(self.currSymbols) == 0:
            return
        currParentName = self.currSubprocName if not isModule else self.currModuleName
        parentNode = self.routineNodesByProcName[currParentName] if not isModule else self.moduleNodesByName[currParentName]
        domainDependantsParentNodes = parentNode.getElementsByTagName("domainDependants")
        if domainDependantsParentNodes == None or len(domainDependantsParentNodes) == 0:
            raise Exception("Unexpected error: No domain dependant parent node found for parent %s where it has been identified before. Parent node: %s. Looking for symbols %s" %(
                currParentName, parentNode.toxml(), str(self.currSymbols)
            ))
        domainDependantsParentNode = domainDependantsParentNodes[0]
        domainDependantEntryNodes = domainDependantsParentNode.getElementsByTagName("entry")
        if domainDependantEntryNodes == None or len(domainDependantEntryNodes) == 0:
            raise Exception("Unexpected error: No domain dependants found for parent %s where they have been identified before. Looking for symbols %s" %(
                currParentName, parentNode.toxml(), str(self.currSymbols)
            ))
        self.entryNodesBySymbolName = {}
        for domainDependantEntryNode in domainDependantEntryNodes:
            self.entryNodesBySymbolName[domainDependantEntryNode.firstChild.nodeValue.strip()] = domainDependantEntryNode

    def storeCurrentSymbolAttributes(self, isModule=False):
        #store our symbol informations to the xml
        for symbol in self.currSymbols:
            if symbol.isModuleSymbol and isModule == False:
                continue
            entryNode = self.entryNodesBySymbolName[symbol.name]
            if not entryNode:
                raise Exception("Unexpected error: symbol named %s not expected" %(symbol.name))
            symbol.storeDomainDependantEntryNodeAttributes(entryNode)

    def processModuleEndMatch(self, moduleEndMatch):
        #get handles to currently active symbols -> temporarily save the handles
        self.processSymbolAttributes(isModule=True)
        if self.debugPrint:
            sys.stderr.write("exiting module %s. Storing informations for symbols %s\n" %(self.currModuleName, str(self.currSymbols)))
        #finish parsing -> superclass destroys handles
        super(H90XMLSymbolDeclarationExtractor, self).processModuleEndMatch(moduleEndMatch)
        #store our symbol informations to the xml
        self.storeCurrentSymbolAttributes(isModule=True)
        #throw away our handles
        self.entryNodesBySymbolName = {}
        self.currSymbols = []


    def processProcEndMatch(self, subProcEndMatch):
        #get handles to currently active symbols -> temporarily save the handles
        self.processSymbolAttributes()
        if self.debugPrint:
            sys.stderr.write("exiting procedure %s. Storing informations for symbols %s\n" %(self.currSubprocName, str(self.currSymbols)))
        #finish parsing -> superclass destroys handles
        super(H90XMLSymbolDeclarationExtractor, self).processProcEndMatch(subProcEndMatch)
        #store our symbol informations to the xml
        self.storeCurrentSymbolAttributes()
        #throw away our handles
        self.entryNodesBySymbolName = {}
        self.currSymbols = []


class H90toF90Printer(H90CallGraphAndSymbolDeclarationsParser):
    currSubroutineImplementationNeedsToBeCommented = False
    currRoutineIsCallingParallelRegion = False
    currCalleeNode = None
    additionalSymbolsByCalleeName = {}
    currAdditionalSubroutineParameters = []
    currAdditionalCompactedSubroutineParameters = []
    symbolsPassedInCurrentCallByName = {}
    currParallelIterators = []
    intentPattern = None
    dimensionPattern = None
    tab_insideSub = "\t\t"
    tab_outsideSub = "\t"
    implementationsByTemplateName = None
    codeSanitizer = None
    stateBeforeBranch = None
    currParallelRegionRelationNode = None
    currParallelRegionTemplateNode = None

    def __init__(self, cgDoc, implementationsByTemplateName):
        self.implementationsByTemplateName = implementationsByTemplateName
        self.currRoutineIsCallingParallelRegion = False
        self.currSubroutineImplementationNeedsToBeCommented = False
        self.symbolsPassedInCurrentCallByName = {}
        self.additionalSymbolsByCalleeName = {}
        self.currParallelIterators = []
        self.currentLine = ""
        self.currCalleeNode = None
        self.currAdditionalSubroutineParameters = []
        self.currAdditionalCompactedSubroutineParameters = []
        self.codeSanitizer = FortranCodeSanitizer()
        self.currParallelRegionRelationNode = None
        self.currParallelRegionTemplateNode = None
        super(H90toF90Printer, self).__init__(cgDoc)

    @property
    def implementation(self):
        implementation = self.implementationsByTemplateName.get(self.currTemplateName)
        if implementation == None:
            implementation = self.implementationsByTemplateName.get('default')
        if implementation == None:
            raise Exception("no default implementation defined")
        return implementation

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
        if len(templates) > 1 and self.implementation.multipleParallelRegionsPerSubroutineAllowed != True:
            raise Exception("Unexpected: more than one parallel region templates found for subroutine '%s' containing a parallelRegion directive \
This is not allowed for implementations using %s.\
                " %(self.currSubprocName, type(self.implementation).__name__)
            )

        implementationAttr = getattr(self, 'implementation')
        functionAttr = getattr(implementationAttr, implementationFunctionName)
        self.prepareLine(functionAttr(self.currParallelRegionTemplateNode, self.branchAnalyzer.level), self.tab_insideSub)

    def processSymbolMatchAndGetAdjustedLine(self, line, symbolMatch, symbol, isInsideSubroutineCall, isPointerAssignment):
        def getAccessorsAndRemainder(accessorString):
            currBracketAnalyzer = BracketAnalyzer()
            accessors = []
            symbol_access_match = self.patterns.symbolAccessPattern.match(accessorString)
            if not symbol_access_match:
                return accessors, accessorString
            work = symbol_access_match.group(1)
            if len(work) < 1:
                raise Exception("unmatched opening bracket: %s" %(accessorString))
            bracketLevel = 0
            while len(work) > 0 and bracketLevel >= 0:
                currAccessor, work, bracketLevel = currBracketAnalyzer.splitAfterCharacterOnSameLevelOrClosingBrackets(work, ',')
                work = work.strip()
                currAccessor = currAccessor.strip()
                #just some sanity checks
                if bracketLevel > 0:
                    raise Exception("There was a problem in the bracket analysis for the following string: %s" %(accessorString))
                elif bracketLevel == 0 and work == "" or bracketLevel < 0 and currAccessor[-1] != ')':
                    raise Exception("Closing bracket expected but none found in accessor string: %s" %(accessorString))
                if bracketLevel == 0 and len(work) > 0 and work[0] == ',':
                    work = work[1:len(work)]
                if len(currAccessor) > 0 and currAccessor[-1] == ',' or len(currAccessor) > 0 and currAccessor[-1] == ')' and bracketLevel < 0:
                    currAccessor = currAccessor[0:len(currAccessor)-1]
                if currAccessor == "" and bracketLevel < 0:
                    break
                if currAccessor == "":
                    raise Exception("Invalid empty accessor. Analyzed string: %s; Accessors so far: %s; Remainder: %s" %(accessorString, str(accessors), work))
                accessors.append(currAccessor.strip())
            return accessors, work

        #match the symbol's postfix again in the current given line. (The prefix could have changed from the last match.)
        postfix = symbolMatch.group(3)
        postfixEscaped = re.escape(postfix)
        accessors, postfix = getAccessorsAndRemainder(postfix)

        if symbol.domains \
        and len(symbol.domains) > 0 \
        and not isInsideSubroutineCall \
        and not isPointerAssignment \
        and not symbol.isModuleSymbol \
        and not symbol.isHostSymbol \
        and self.state != "inside_parallelRegion" \
        and self.routineNodesByProcName[self.currSubprocName].getAttribute("parallelRegionPosition") != "outside" \
        and len(accessors) != 0:
            sys.stderr.write("WARNING: Dependant symbol %s accessed with accessor domains (%s) outside of a parallel region or subroutine call in subroutine %s(%s:%i)\n" \
                %(symbol.name, str(accessors), self.currSubprocName, self.fileName, self.lineNo)
            )

        accessPatternChangeRequired = False
        presentPattern = r"(.*?present\s*\(\s*)" + re.escape(symbol.deviceName()) + postfixEscaped + r"\s*"
        currMatch = re.match(presentPattern, line, re.IGNORECASE)
        if not currMatch:
            pattern1 = r"(.*?(?:\W|^))" + re.escape(symbol.deviceName()) + postfixEscaped + r"\s*"
            currMatch = re.match(pattern1, line, re.IGNORECASE)
            accessPatternChangeRequired = True
            if not currMatch:
                pattern2 = r"(.*?(?:\W|^))" + re.escape(symbol.name) + postfixEscaped + r"\s*"
                currMatch = re.match(pattern2, line, re.IGNORECASE)
                if not currMatch:
                    raise Exception(\
                        "Symbol %s is accessed in an unexpected way. Note: '_d' postfix is reserved for internal use. Cannot match one of the following patterns: \npattern1: '%s'\npattern2: '%s'" \
                        %(symbol.name, pattern1, pattern2))
        prefix = currMatch.group(1)
        if accessPatternChangeRequired:
            numOfIndependentDomains = len(symbol.domains) - symbol.numOfParallelDomains
            offsets = []
            if len(accessors) != numOfIndependentDomains and len(accessors) != len(symbol.domains) and len(accessors) != 0:
                raise Exception("Unexpected array access for symbol %s (%s): Please use either %i (number of parallel independant dimensions) \
    or %i (number of declared dimensions for this array) or zero accessors. Symbol Domains: %s; Symbol Init Level: %i; Parallel Region Position: %s; Parallel Active: %s; Symbol template:\n%s\n" %(
                    symbol.name,
                    str(accessors),
                    numOfIndependentDomains,
                    len(symbol.domains),
                    str(symbol.domains),
                    symbol.initLevel,
                    str(symbol.parallelRegionPosition),
                    symbol.parallelActiveDims,
                    symbol.template.toxml()
                ))
            if len(accessors) == 0 and (isInsideSubroutineCall or isPointerAssignment):
                for i in range(numOfIndependentDomains):
                    offsets.append(":")
            else:
                offsets += accessors

            iterators = self.currParallelIterators
            if isInsideSubroutineCall:
                calleeNode = self.routineNodesByProcName.get(self.currCalleeName)
                if calleeNode and calleeNode.getAttribute("parallelRegionPosition") != "outside":
                    iterators = []
        symbol_access = None
        if isPointerAssignment \
        or not accessPatternChangeRequired \
        or ( \
            self.state != "inside_parallelRegion" \
            and not isInsideSubroutineCall \
            and not isPointerAssignment \
            and not symbol.isModuleSymbol \
            and not symbol.isHostSymbol \
            and len(accessors) == 0 \
        ):
            symbol_access = symbol.deviceName()
        else:
            symbol_access = symbol.accessRepresentation(
                iterators,
                offsets,
                self.currParallelRegionTemplateNode,
                inside_subroutine_call=isInsideSubroutineCall
            )
        if self.debugPrint:
            sys.stderr.write("symbol %s on line %i rewritten to %s\n" %(str(symbol), self.lineNo, symbol_access))
        return (prefix + symbol_access + postfix).rstrip() + "\n"

    def processSymbolsAndGetAdjustedLine(self, line, isInsideSubroutineCall):
        isPointerAssignment = self.patterns.pointerAssignmentPattern.match(line) != None
        symbolNames = self.currSymbolsByName.keys()
        adjustedLine = line
        for symbolName in symbolNames:
            symbol = self.currSymbolsByName[symbolName]
            symbolWasMatched = False
            lineSections = []
            work = adjustedLine
            nextMatch = symbol.namePattern.match(work)
            while nextMatch:
                symbolWasMatched = True
                prefix = nextMatch.group(1)
                lineSections.append(prefix)
                postfix = nextMatch.group(3)
                processed = self.processSymbolMatchAndGetAdjustedLine(work, nextMatch, symbol, isInsideSubroutineCall, isPointerAssignment)
                adjustedMatch = symbol.namePattern.match(processed)
                if not adjustedMatch:
                    raise Exception("Unexpected error: symbol %s can't be matched again after adjustment. Adjusted portion: %s" %(symbol.name, processed))
                lineSections.append(adjustedMatch.group(2))
                work = adjustedMatch.group(3)
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
        if self.debugPrint:
            sys.stderr.write("In subroutine %s: Processing subroutine call to %s, parallel region position: %s\n" \
                %(self.currSubprocName, self.currCalleeName, parallelRegionPosition) \
            )
        if self.currCalleeNode and parallelRegionPosition == "within":
            parallelRegionTemplates = self.parallelRegionTemplatesByProcName.get(self.currCalleeName)
            if parallelRegionTemplates == None or len(parallelRegionTemplates) == 0:
                raise Exception("Unexpected error: No parallel region templates found for subroutine %s" %(self.currCalleeName))
            adjustedLine = self.implementation.kernelCallPreparation(parallelRegionTemplates[0], calleeNode=self.currCalleeNode)
            adjustedLine = adjustedLine + "call " + self.currCalleeName + " " + self.implementation.kernelCallConfig()

        # if self.currCalleeNode \
        # and getRoutineNodeInitStage(self.currCalleeNode) == RoutineNodeInitStage.DIRECTIVES_WITHOUT_PARALLELREGION_POSITION:
        #     #special case. see also processBeginMatch.
        #     self.prepareLine("! " + subProcCallMatch.group(0), "")
        #     self.symbolsPassedInCurrentCallByName = {}
        #     self.currCalleeNode = None
        #     return

        arguments = subProcCallMatch.group(2)
        paramListMatch = self.patterns.subprocFirstLineParameterListPattern.match(arguments)
        if not paramListMatch and len(arguments.strip()) > 0:
            raise Exception("Subprocedure arguments without enclosing brackets. This is invalid in Hybrid Fortran")
        additionalSymbolsSeparated = self.additionalSymbolsByCalleeName.get(self.currCalleeName)
        toBeCompacted = []
        if additionalSymbolsSeparated and len(additionalSymbolsSeparated) > 1:
            toBeCompacted, declarationPrefix, otherImports = self.listCompactedSymbolsAndDeclarationPrefixAndOtherImports(additionalSymbolsSeparated[0])
        compactedArrayList = []
        if len(toBeCompacted) > 0:
            compactedArrayName = "hfimp_%s" %(self.currCalleeName)
            compactedArray = FrameworkArray(compactedArrayName, declarationPrefix, domains=[("hfauto", str(len(toBeCompacted)))], isOnDevice=True)
            compactedArrayList = [compactedArray]
        if additionalSymbolsSeparated:
            additionalSymbols = sorted(additionalSymbolsSeparated[1] + otherImports + compactedArrayList)
        else:
            additionalSymbols = []
        if len(additionalSymbols) > 0:
            adjustedLine = adjustedLine + "( &\n"
        else:
            adjustedLine = adjustedLine + "("
        symbolNum = 0
        bridgeStr = ", & !additional parameter inserted by framework\n" + self.tab_insideSub + "& "
        for symbol in additionalSymbols:
            adjustedLine = adjustedLine + symbol.automaticName()
            if symbolNum < len(additionalSymbols) - 1 or paramListMatch:
                adjustedLine = adjustedLine + bridgeStr
            symbolNum = symbolNum + 1
        if paramListMatch:
            adjustedLine = adjustedLine + self.processSymbolsAndGetAdjustedLine(paramListMatch.group(2), isInsideSubroutineCall=True)
        else:
            adjustedLine = adjustedLine + ")\n"

        callPreparationForSymbols = ""
        callPostForSymbols = ""
        if self.currCalleeNode and self.currCalleeNode.getAttribute("parallelRegionPosition") == "within":
            if self.state != "inside_subroutine_call":
                currSubprocNode = self.routineNodesByProcName.get(self.currSubprocName)
                callPreparationForSymbols = "".join([
                    self.implementation.callPreparationForPassedSymbol(currSubprocNode, symbol)
                    for symbol in self.symbolsPassedInCurrentCallByName.values()
                ])
                callPostForSymbols = "".join([
                    self.implementation.callPostForPassedSymbol(currSubprocNode, symbol)
                    for symbol in self.symbolsPassedInCurrentCallByName.values()
                ])
                adjustedLine = self.processCallPostAndGetAdjustedLine(adjustedLine)

        if self.state != "inside_subroutine_call":
            self.symbolsPassedInCurrentCallByName = {}
            self.currCalleeNode = None

        self.prepareLine(callPreparationForSymbols + adjustedLine + callPostForSymbols, self.tab_insideSub)

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

    def processTemplateMatch(self, templateMatch):
        super(H90toF90Printer, self).processTemplateMatch(templateMatch)
        self.prepareLine("","")

    def processTemplateEndMatch(self, templateEndMatch):
        super(H90toF90Printer, self).processTemplateEndMatch(templateEndMatch)
        self.prepareLine("","")

    def processBranchMatch(self, branchMatch):
        super(H90toF90Printer, self).processBranchMatch(branchMatch)
        self.prepareLine("","")

    def processModuleBeginMatch(self, moduleBeginMatch):
        super(H90toF90Printer, self).processModuleBeginMatch(moduleBeginMatch)
        self.implementation.processModuleBegin(self.currModuleName)

    def processModuleEndMatch(self, moduleEndMatch):
        super(H90toF90Printer, self).processModuleEndMatch(moduleEndMatch)
        self.implementation.processModuleEnd()

    def processProcBeginMatch(self, subProcBeginMatch):
        super(H90toF90Printer, self).processProcBeginMatch(subProcBeginMatch)

        subprocName = subProcBeginMatch.group(1)
        routineNode = self.routineNodesByProcName.get(subprocName)
        if not routineNode:
            raise Exception("no definition found for routine '%s'" %(subprocName))

        #build list of additional subroutine parameters
        #(parameters that the user didn't specify but that are necessary based on the features of the underlying technology
        # and the symbols declared by the user, such us temporary arrays and imported symbols)
        toBeCompacted, declarationPrefix, otherImports = self.listCompactedSymbolsAndDeclarationPrefixAndOtherImports(
            self.implementation.extractListOfAdditionalSubroutineSymbols(routineNode, self.currSymbolsByName)
        )
        compactedArrayList = []
        if len(toBeCompacted) > 0:
            compactedArrayName = "hfimp_%s" %(subprocName)
            compactedArray = FrameworkArray(compactedArrayName, declarationPrefix, domains=[("hfauto", str(len(toBeCompacted)))], isOnDevice=True)
            compactedArrayList = [compactedArray]
        self.currAdditionalSubroutineParameters = sorted(otherImports + compactedArrayList)
        self.currAdditionalCompactedSubroutineParameters = sorted(toBeCompacted)
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

    def processProcExitPoint(self, line, is_subroutine_end):
        self.prepareLine(
            self.implementation.subroutineExitPoint(
                self.currSymbolsByName.values(), self.currRoutineIsCallingParallelRegion, is_subroutine_end
            ) + line,
            self.tab_outsideSub
        )

    def processProcEndMatch(self, subProcEndMatch):
        self.processProcExitPoint(subProcEndMatch.group(0), is_subroutine_end=True)
        self.currRoutineIsCallingParallelRegion = False
        self.additionalSymbolsByCalleeName = {}
        self.currAdditionalSubroutineParameters = []
        self.currAdditionalCompactedSubroutineParameters = []
        self.currSubroutineImplementationNeedsToBeCommented = False
        super(H90toF90Printer, self).processProcEndMatch(subProcEndMatch)

    def processParallelRegionMatch(self, parallelRegionMatch):
        self.prepareLine("", "")
        super(H90toF90Printer, self).processParallelRegionMatch(parallelRegionMatch)
        if self.debugPrint:
            sys.stderr.write("...parallel region starts on line %i with active symbols %s\n" \
                %(self.lineNo, str(self.currSymbolsByName.values())) \
            )
        self.prepareActiveParallelRegion('parallelRegionBegin')

    def processParallelRegionEndMatch(self, parallelRegionEndMatch):
        self.prepareLine("", "")
        super(H90toF90Printer, self).processParallelRegionEndMatch(parallelRegionEndMatch)
        self.prepareActiveParallelRegion('parallelRegionEnd')
        self.currParallelIterators = []
        self.currParallelRegionTemplateNode = None
        self.currParallelRegionRelationNode = None

    def processDomainDependantMatch(self, domainDependantMatch):
        super(H90toF90Printer, self).processDomainDependantMatch(domainDependantMatch)
        self.prepareLine("", "")

    def processDomainDependantEndMatch(self, domainDependantEndMatch):
        super(H90toF90Printer, self).processDomainDependantEndMatch(domainDependantEndMatch)
        self.prepareLine("", "")

    def processNoMatch(self):
        super(H90toF90Printer, self).processNoMatch()
        self.prepareLine(str(self.currentLine), "")

    def listCompactedSymbolsAndDeclarationPrefixAndOtherImports(self, additionalImports):
        toBeCompacted = []
        otherImports = []
        declarationPrefix = None
        for symbol in additionalImports:
            declType = symbol.declarationType()

            # compact the imports of real type. Background: Experience has shown that too many
            # symbols passed to kernels, such that parameter list > 256Byte, can cause strange behavior. (corruption
            # of parameter list leading to launch failures)
            #-> need to pack all real symbols into an array to make it work reliably for cases where many reals are imported
            # why not integers? because they can be used as array boundaries.
            # Note: currently only a single real type per subroutine is supported for compaction
            if declType in [DeclarationType.IMPORTED_SCALAR, DeclarationType.MODULE_SCALAR] \
            and 'real' in symbol.declarationPrefix.lower() \
            and (declarationPrefix == None \
            or symbol.declarationPrefix.strip().lower() == declarationPrefix.strip().lower()):
                declarationPrefix = symbol.declarationPrefix
                symbol.isCompacted = True
                toBeCompacted.append(symbol)
            else:
                otherImports.append(symbol)
        return toBeCompacted, declarationPrefix, otherImports


    def processInsideDeclarationsState(self, line):
        '''process everything that happens per h90 declaration line'''
        super(H90toF90Printer, self).processInsideDeclarationsState(line)
        routineNode = self.routineNodesByProcName.get(self.currSubprocName)
#         if routineNode \
#         and getRoutineNodeInitStage(routineNode) == RoutineNodeInitStage.DIRECTIVES_WITHOUT_PARALLELREGION_POSITION \
#         and self.implementation.onDevice:
#             #during analysis we've found that this routine had a parallel region directive but now
#             #there is no relation to a parallel region anymore in the callgraph.
#             #This is a special case where the programmer most probably commented out the call to this subroutine
#             #=> go into skipped state where the whole subroutine body is printed out commented (don't compile the body)
#             if self.currSubroutineImplementationNeedsToBeCommented != True:
#                 sys.stderr.write("WARNING: subroutine body for %s is getting commented out by HF - can't determine the position of its parallel region \
# relative to the rest of the program. Please ignore this message if you have intentionally commented out a call to the respective subroutine \n" \
#                     %(self.currSubprocName) \
#                 )
#                 self.currSubroutineImplementationNeedsToBeCommented = True
#             if self.state == "inside_declarations":
#                 self.prepareLine(line, "")
#             else:
#                 self.prepareLine("! " + line, "")
#             return

        if self.state != "inside_declarations" and self.state != "inside_module" and self.state != "inside_subroutine_call":
            additionalDeclarationsStr = ""
            if len(self.additionalSymbolsByCalleeName.keys()) > 0:
                additionalDeclarationsStr = "\n" + self.tab_insideSub + \
                 "! ****** additional symbols inserted by framework to emulate device support of language features\n"

            #########################################################################
            # additional symbols for called kernel                                  #
            #########################################################################
            packedRealSymbolsByCalleeName = {}
            for calleeName in self.additionalSymbolsByCalleeName.keys():
                additionalImports, additionalDeclarations = self.additionalSymbolsByCalleeName[calleeName]
                additionalImportSymbolsByName = {}
                for symbol in additionalImports:
                    additionalImportSymbolsByName[symbol.name] = symbol

                for symbol in additionalDeclarations:
                    declType = symbol.declarationType()
                    if declType != DeclarationType.IMPORTED_SCALAR and declType != DeclarationType.LOCAL_ARRAY:
                        continue

                    #in case the array uses domain sizes in the declaration that are additional symbols themselves
                    #we need to fix them.
                    adjustedDomains = []
                    for (domName, domSize) in symbol.domains:
                        domSizeSymbol = additionalImportSymbolsByName.get(domSize)
                        if domSizeSymbol is None:
                            adjustedDomains.append((domName, domSize))
                            continue
                        adjustedDomains.append((domName, domSizeSymbol.automaticName()))
                    symbol.domains = adjustedDomains

                    additionalDeclarationsStr = additionalDeclarationsStr + \
                        self.implementation.adjustDeclarationForDevice( \
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

                toBeCompacted, declarationPrefix, _ = self.listCompactedSymbolsAndDeclarationPrefixAndOtherImports(additionalImports)
                if len(toBeCompacted) > 0:
                    #TODO: generalize for cases where we don't want this to be on the device (e.g. put this into Implementation class)
                    compactedArrayName = "hfimp_%s" %(calleeName)
                    compactedArray = FrameworkArray(compactedArrayName, declarationPrefix, domains=[("hfauto", str(len(toBeCompacted)))], isOnDevice=True)
                    packedRealSymbolsByCalleeName[calleeName] = toBeCompacted
                    additionalDeclarationsStr = additionalDeclarationsStr + \
                        self.implementation.adjustDeclarationForDevice( \
                            compactedArray.getDeclarationLineForAutomaticSymbol().strip(), \
                            self.patterns, \
                            [compactedArray], \
                            self.currRoutineIsCallingParallelRegion, \
                            self.routineNodesByProcName[self.currSubprocName].getAttribute('parallelRegionPosition') \
                        ).rstrip() + "\n"
                    if self.debugPrint:
                        sys.stderr.write("...In subroutine %s: Symbols %s packed into array %s\n" \
                            %(self.currSubprocName, toBeCompacted, compactedArrayName) \
                        )


            #########################################################################
            # additional symbols for ourselves                                      #
            #########################################################################
            ourSymbolsToAdd = sorted(
                [symbol for symbol in self.currAdditionalSubroutineParameters
                    if symbol.declarationType() in [DeclarationType.IMPORTED_SCALAR,
                        DeclarationType.MODULE_SCALAR,
                        DeclarationType.FRAMEWORK_ARRAY
                    ]
                ] + self.currAdditionalCompactedSubroutineParameters
            )
            for symbol in ourSymbolsToAdd:
                purgeList=[]
                if symbol.isCompacted:
                    purgeList=['intent']
                additionalDeclarationsStr = additionalDeclarationsStr + \
                    self.tab_insideSub + self.implementation.adjustDeclarationForDevice( \
                        self.tab_insideSub + \
                            symbol.getDeclarationLineForAutomaticSymbol(purgeList, self.patterns).strip(), \
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
            additionalDeclarationsStr += self.implementation.declarationEnd( \
                    self.currSymbolsByName.values(), \
                    self.currRoutineIsCallingParallelRegion, \
                    self.routineNodesByProcName[self.currSubprocName], \
                    self.parallelRegionTemplatesByProcName.get(self.currSubprocName))

            #########################################################################
            # additional symbols to be packed into arrays                           #
            #########################################################################
            #TODO: move this into implementation classes
            calleesWithPackedReals = packedRealSymbolsByCalleeName.keys()
            for calleeName in calleesWithPackedReals:
                for idx, symbol in enumerate(sorted(packedRealSymbolsByCalleeName[calleeName])):
                    additionalDeclarationsStr = additionalDeclarationsStr + "hfimp_%s(%i) = %s\n" %(calleeName, idx+1, symbol.automaticName())

            #########################################################################
            # additional symbols to be unpacked                                     #
            #########################################################################
            #TODO: move this into implementation classes
            for idx, symbol in enumerate(self.currAdditionalCompactedSubroutineParameters):
                additionalDeclarationsStr = additionalDeclarationsStr + "%s = hfimp_%s(%i)\n" %(symbol.deviceName(), self.currSubprocName, idx+1)

            self.prepareLine(additionalDeclarationsStr, self.tab_insideSub)

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
            self.state = 'inside_module'
            self.currSubprocName = None
            return

        if (self.patterns.earlyReturnPattern.match(str(line))):
            self.processProcExitPoint(line, is_subroutine_end=False)
            return

        if self.currSubroutineImplementationNeedsToBeCommented:
            self.prepareLine("! " + line, "")
            return

        parallelRegionMatch = self.patterns.parallelRegionPattern.match(str(line))
        if (parallelRegionMatch) \
        and self.routineNodesByProcName[self.currSubprocName].getAttribute('parallelRegionPosition') == "within":
            templateRelations = self.parallelRegionTemplateRelationsByProcName.get(self.currSubprocName)
            if templateRelations == None or len(templateRelations) == 0:
                raise Exception("Unexpected Error: No parallel region template relation found for this region.")
            for templateRelation in templateRelations:
                startLine = templateRelation.getAttribute("startLine")
                if startLine in [None, '']:
                    continue
                startLineInt = 0
                try:
                    startLineInt = int(startLine)
                except ValueError:
                    raise Exception("Unexpected Error: Invalid startLine definition for parallel region template relation: %s\n. All active template relations: %s\nRoutine node: %s" %(
                        templateRelation.toxml(),
                        [templateRelation.toxml() for templateRelation in templateRelations],
                        self.routineNodesByProcName[self.currSubprocName].toprettyxml()
                    ))
                if startLineInt == self.lineNo:
                    self.currParallelRegionRelationNode = templateRelation
                    break
            else:
                raise Exception("Unexpected Error: No parallel region template relation was matched for the current linenumber.")
            if self.debugPrint:
                sys.stderr.write("parallel region detected on line %i with template relation %s\n" \
                    %(self.lineNo, self.currParallelRegionRelationNode.toxml()) \
                )
            templates = self.parallelRegionTemplatesByProcName.get(self.currSubprocName)
            if templates == None or len(templates) == 0:
                raise Exception("Unexpected Error: No parallel region template found for this region.")
            activeTemplateID = self.currParallelRegionRelationNode.getAttribute("id")
            for template in templates:
                if template.getAttribute("id") == activeTemplateID:
                    self.currParallelRegionTemplateNode = template
                    break
            else:
                raise Exception("Unexpected Error: No parallel region template has matched the active template ID.")
            self.currParallelIterators = self.implementation.getIterators(self.currParallelRegionTemplateNode)
            if len(self.currParallelIterators) > 0:
                self.processParallelRegionMatch(parallelRegionMatch)
                self.state = 'inside_parallelRegion'
            else:
                self.prepareLine("","")
            return
        elif parallelRegionMatch:
            #this parallel region does not apply to us
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

        adjustedLine = ""
        loopMatch = self.patterns.loopPattern.match(str(line))
        if loopMatch:
            adjustedLine += self.implementation.loopPreparation().strip() + '\n'
        adjustedLine += self.processSymbolsAndGetAdjustedLine(line, isInsideSubroutineCall=False)
        self.prepareLine(adjustedLine, self.tab_insideSub)

    def processInsideSubroutineCall(self, line):
        super(H90toF90Printer, self).processInsideSubroutineCall(line)
        adjustedLine = ""

        # if self.currCalleeNode and getRoutineNodeInitStage(self.currCalleeNode) == RoutineNodeInitStage.DIRECTIVES_WITHOUT_PARALLELREGION_POSITION:
        #     #special case. see also processBeginMatch.
        #     adjustedLine = "! " + line
        # else:
        adjustedLine = self.processSymbolsAndGetAdjustedLine(line, isInsideSubroutineCall=True)

        if self.currCalleeNode and self.state != "inside_subroutine_call":
            adjustedLine = self.processCallPostAndGetAdjustedLine(adjustedLine)
        self.prepareLine(adjustedLine.rstrip() + "\n", "")

    def processInsideDomainDependantRegionState(self, line):
        super(H90toF90Printer, self).processInsideDomainDependantRegionState(line)
        self.prepareLine("", "")

    def processInsideModuleDomainDependantRegionState(self, line):
        super(H90toF90Printer, self).processInsideModuleDomainDependantRegionState(line)
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
        sys.stdout.write(self.implementation.filePreparation(fileName))
        super(H90toF90Printer, self).processFile(fileName)

    #TODO: remove tab argument everywhere
    def prepareLine(self, line, tab):
        self.currentLine = self.codeSanitizer.sanitizeLines(line)
        if self.debugPrint:
            sys.stderr.write("[%s]:%i:%s\n" \
                %(self.state,self.lineNo,self.currentLine) \
            )
