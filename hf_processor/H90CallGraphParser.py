#!/usr/bin/python
# -*- coding: UTF-8 -*-

# Copyright (C) 2015 Michel Müller, Tokyo Institute of Technology

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
from GeneralHelper import UsageError, BracketAnalyzer, findRightMostOccurrenceNotInsideQuotes, stripWhitespace, enum
from H90SymbolDependencyGraphAnalyzer import SymbolDependencyAnalyzer
from H90Symbol import *
from H90RegExPatterns import H90RegExPatterns
import os
import sys
import fileinput
import re
import uuid
import pdb
import traceback
import logging

currFile = None
currLineNo = None

class FortranRoutineArgumentParser:
    arguments = None

    def __init__(self):
        self.arguments = []

    def __repr__(self):
        return "[ArgParser: %s]" %(str(self.arguments))

    def processString(self, string, patterns):
        argumentMatch = patterns.argumentPattern.match(string)
        if not argumentMatch:
            return
        currBracketAnalyzer = BracketAnalyzer()
        arguments, _ = currBracketAnalyzer.getListOfArgumentsInOpenedBracketsAndRemainder(argumentMatch.group(1))
        self.arguments = arguments

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
        self.openACCLinePattern = re.compile(r'\s*\!\$ACC.*', re.IGNORECASE)
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
            remainder = codeLine
            previousLineLength = len(remainder)
            blankPos = -1
            remainderContainsLineContinuation = False
            while len(remainder) > howManyCharsPerLine - len(lineSep):
                isOpenMPDirectiveLine = self.openMPLinePattern.match(remainder) != None
                isOpenACCDirectiveLine = self.openACCLinePattern.match(remainder) != None
                if commentChar in remainder and not isOpenMPDirectiveLine and not isOpenACCDirectiveLine:
                    commentPos = remainder.find(commentChar)
                    if commentPos <= howManyCharsPerLine:
                        break
                #find a blank that's NOT within a quoted string
                searchString = remainder
                prevLineContinuation = ""
                if remainderContainsLineContinuation and (isOpenMPDirectiveLine or isOpenACCDirectiveLine):
                    searchString = remainder[7:]
                    prevLineContinuation = remainder[:7]
                elif remainderContainsLineContinuation:
                    searchString = remainder[2:]
                    prevLineContinuation = remainder[:2]
                startOffset = 0
                while len(searchString) + len(prevLineContinuation) > howManyCharsPerLine - len(lineSep) + startOffset:
                    blankPos = findRightMostOccurrenceNotInsideQuotes(' ', searchString, rightStartAt=howManyCharsPerLine - len(lineSep) + startOffset)
                    startOffset += 5 #if nothing is possible to break up it's better to go a little bit over the limit, often the compiler will still cope
                    if blankPos >= 1:
                        break
                if blankPos < 1:
                    currLine = remainder
                    remainder = ""
                else:
                    currLine = prevLineContinuation + searchString[:blankPos] + lineSep
                    if blankPos >= 1 and isOpenMPDirectiveLine:
                        remainder = '!$OMP& ' + searchString[blankPos:]
                        remainderContainsLineContinuation = True
                    elif blankPos >= 1 and isOpenACCDirectiveLine:
                        remainder = '!$acc& ' + searchString[blankPos:]
                        remainderContainsLineContinuation = True
                    elif blankPos >= 1:
                        remainder = '& ' + searchString[blankPos:]
                        remainderContainsLineContinuation = True
                sanitizedCodeLines.append(currLine)
                if blankPos < 1 or len(remainder) >= previousLineLength:
                    #blank not found or at beginning of line
                    #-> bail out in order to avoid infinite loop - just keep the line as it was.
                    logging.warning(
                        "The following line could not be broken up for Fortran compatibility - no suitable spaces found: %s (remainder: %s)\n" %(
                            currLine,
                            remainder
                        ),
                        extra={"hfLineNo":currLineNo, "hfFile":currFile}
                    )
                    break
                previousLineLength = len(remainder)
            if toBeCommented:
                currLine = commentChar + " " + currLine
            if remainder != "":
                sanitizedCodeLines.append(remainder)

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
    state = 'none'
    currSubprocName = None
    currModuleName = None
    currArgumentParser = None
    currCalleeName = None
    currArguments = None
    patterns = None
    currentLine = None
    branchAnalyzer = None
    currTemplateName = None
    stateSwitch = None
    currSymbolsByName = None

    def __init__(self):
        self.patterns = H90RegExPatterns.Instance()
        self.state = "none"
        self.currCalleeName = None
        self.currArguments = None
        self.currModuleName = None
        self.currSymbolsByName = {}
        self.branchAnalyzer = BracketAnalyzer(
            r'^\s*if\s*\(|^\s*select\s+case',
            r'^\s*end\s+if|^\s*end\s+select',
            pass_in_regex_pattern=True
        )
        self.stateSwitch = {
           'none': self.processNoneState,
           'inside_module': self.processInsideModuleState,
           'inside_moduleDomainDependantRegion': self.processInsideModuleDomainDependantRegionState,
           'inside_declarations': self.processInsideDeclarationsState,
           'inside_parallelRegion': self.processInsideParallelRegionState,
           'inside_domainDependantRegion': self.processInsideDomainDependantRegionState,
           'inside_subroutine_body': self.processInsideSubroutineBodyState,
           'inside_branch': self.processInsideBranch,
           'inside_ignore': self.processInsideIgnore
         }
        super(H90CallGraphParser, self).__init__()

    def processCallMatch(self, subProcCallMatch):
        if (not subProcCallMatch.group(1) or subProcCallMatch.group(1) == ''):
            raise UsageError("subprocedure call without matching subprocedure name")
        self.currArgumentParser = FortranRoutineArgumentParser()
        self.currArgumentParser.processString(subProcCallMatch.group(0), self.patterns)
        self.currArguments = self.currArgumentParser.arguments
        self.currCalleeName = subProcCallMatch.group(1)
        return

    def processCallPost(self):
        self.currArguments = None
        self.currArgumentParser = None
        self.currCalleeName = None

    def processBranchMatch(self, branchMatch):
        return

    def processProcBeginMatch(self, subProcBeginMatch):
        logging.debug('entering %s' %(subProcBeginMatch.group(1)), extra={"hfLineNo":currLineNo, "hfFile":currFile})
        return

    def processProcEndMatch(self, subProcEndMatch):
        logging.debug('exiting subprocedure', extra={"hfLineNo":currLineNo, "hfFile":currFile})
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
        settingText, remainder = settingBracketAnalyzer.getTextWithinBracketsAndRemainder(textAfterSettingName)
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
        branchMatch = self.patterns.branchPattern.match(str(line))
        if branchMatch:
            self.processBranchMatch(branchMatch)
        elif templateMatch:
            self.processTemplateMatch(templateMatch)
        elif templateEndMatch:
            self.processTemplateEndMatch(templateEndMatch)
        elif moduleBeginMatch:
            self.currModuleName = moduleBeginMatch.group(1)
            self.state = 'inside_module'
            self.processModuleBeginMatch(moduleBeginMatch)
        elif subProcBeginMatch:
            raise UsageError("please put this Hybrid Fortran subroutine into a module")
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
        branchMatch = self.patterns.branchPattern.match(str(line))

        if branchMatch:
            self.processBranchMatch(branchMatch)
        elif templateMatch:
            self.processTemplateMatch(templateMatch)
        elif templateEndMatch:
            self.processTemplateEndMatch(templateEndMatch)
        elif domainDependantMatch:
            if self.state == "inside_branch":
                self.stateBeforeBranch = 'inside_moduleDomainDependantRegion'
            else:
                self.state = 'inside_moduleDomainDependantRegion'
            self.processDomainDependantMatch(domainDependantMatch)
        elif moduleEndMatch:
            self.processModuleEndMatch(moduleEndMatch)
            self.currModuleName = None
            if self.state == "inside_branch":
                self.stateBeforeBranch = 'none'
            else:
                self.state = 'none'
        elif subProcBeginMatch:
            if (not subProcBeginMatch.group(1) or subProcBeginMatch.group(1) == ''):
                raise UsageError("subprocedure begin without matching subprocedure name")
            self.currSubprocName = subProcBeginMatch.group(1)
            self.currArgumentParser = FortranRoutineArgumentParser()
            self.currArgumentParser.processString(subProcBeginMatch.group(0), self.patterns)
            self.currArguments = self.currArgumentParser.arguments
            self.processProcBeginMatch(subProcBeginMatch)
            if self.state == "inside_branch":
                self.stateBeforeBranch = 'inside_declarations'
            else:
                self.state = 'inside_declarations'
            specificationsBeginHere = True
            self.processSubprocStartPost()
        elif (self.patterns.subprocEndPattern.match(str(line))):
            raise UsageError("end subprocedure without matching begin subprocedure")
        else:
            self.processNoMatch()
        if specificationsBeginHere:
            self.processSpecificationBeginning()

    def processSubprocStartPost(self):
        return

    def processInsideDeclarationsState(self, line):
        subProcCallMatch = self.patterns.subprocCallPattern.match(str(line))
        parallelRegionMatch = self.patterns.parallelRegionPattern.match(str(line))
        domainDependantMatch = self.patterns.domainDependantPattern.match(str(line))
        subProcEndMatch = self.patterns.subprocEndPattern.match(str(line))
        templateMatch = self.patterns.templatePattern.match(str(line))
        templateEndMatch = self.patterns.templateEndPattern.match(str(line))
        branchMatch = self.patterns.branchPattern.match(str(line))

        if branchMatch:
            self.processBranchMatch(branchMatch)
        elif (domainDependantMatch):
            if self.state == "inside_branch":
                self.stateBeforeBranch = 'inside_domainDependantRegion'
            else:
                self.state = 'inside_domainDependantRegion'
            self.processDomainDependantMatch(domainDependantMatch)
        elif subProcCallMatch:
            self.processCallMatch(subProcCallMatch)
            if (self.state == "inside_branch" and self.stateBeforeBranch != 'inside_subroutine_call') or (self.state != "inside_branch" and self.state != 'inside_subroutine_call'):
                self.processCallPost()
        elif subProcEndMatch:
            self.processProcEndMatch(subProcEndMatch)
            if self.state == "inside_branch":
                self.stateBeforeBranch = 'inside_module'
            else:
                self.state = 'inside_module'
            self.currSubprocName = None
        elif parallelRegionMatch:
            raise UsageError("parallel region without parallel dependants")
        elif (self.patterns.subprocBeginPattern.match(str(line))):
            raise UsageError("subprocedure within subprocedure not allowed")
        elif templateMatch:
            raise UsageError("template directives are only allowed outside of subroutines")
        elif templateEndMatch:
            raise UsageError("template directives are only allowed outside of subroutines")
        if self.state != "inside_declarations" and not (self.state == "inside_branch" and self.stateBeforeBranch == "inside_declarations"):
            self.currArgumentParser = None
            self.currArguments = None

    def processInsideSubroutineBodyState(self, line):
        #note: Branches (@if statements) are ignored here, we want to keep analyzing their statements for callgraphs.
        domainDependantMatch = self.patterns.domainDependantPattern.match(str(line))
        subProcCallMatch = self.patterns.subprocCallPattern.match(str(line))
        parallelRegionMatch = self.patterns.parallelRegionPattern.match(str(line))
        domainDependantMatch = self.patterns.domainDependantPattern.match(str(line))
        subProcEndMatch = self.patterns.subprocEndPattern.match(str(line))
        templateMatch = self.patterns.templatePattern.match(str(line))
        templateEndMatch = self.patterns.templateEndPattern.match(str(line))
        branchMatch = self.patterns.branchPattern.match(str(line))

        if branchMatch:
            self.processBranchMatch(branchMatch)
        elif domainDependantMatch:
            if self.state == "inside_branch":
                self.stateBeforeBranch = 'inside_domainDependantRegion'
            else:
                self.state = 'inside_domainDependantRegion'
            self.processDomainDependantMatch(domainDependantMatch)
        elif subProcCallMatch:
            self.processCallMatch(subProcCallMatch)
            if (self.state == "inside_branch" and self.stateBeforeBranch != 'inside_subroutine_call') or (self.state != "inside_branch" and self.state != 'inside_subroutine_call'):
                self.processCallPost()
        elif subProcEndMatch:
            self.processProcEndMatch(subProcEndMatch)
            if self.state == "inside_branch":
                self.stateBeforeBranch = 'inside_module'
            else:
                self.state = 'inside_module'
            self.currSubprocName = None
        elif parallelRegionMatch:
            self.processParallelRegionMatch(parallelRegionMatch)
            self.state = 'inside_parallelRegion'
        elif self.patterns.subprocBeginPattern.match(str(line)):
            raise UsageError("subprocedure within subprocedure not allowed")
        elif templateMatch:
            raise UsageError("template directives are only allowed outside of subroutines")
        elif templateEndMatch:
            raise UsageError("template directives are only allowed outside of subroutines")

    def processInsideParallelRegionState(self, line):
        subProcCallMatch = self.patterns.subprocCallPattern.match(str(line))
        parallelRegionEndMatch = self.patterns.parallelRegionEndPattern.match(str(line))
        templateMatch = self.patterns.templatePattern.match(str(line))
        templateEndMatch = self.patterns.templateEndPattern.match(str(line))
        branchMatch = self.patterns.branchPattern.match(str(line))

        newState = None
        if branchMatch:
            self.processBranchMatch(branchMatch)
        elif subProcCallMatch:
            self.processCallMatch(subProcCallMatch)
            if (self.state == "inside_branch" and self.stateBeforeBranch != 'inside_subroutine_call') or (self.state != "inside_branch" and self.state != 'inside_subroutine_call'):
                self.processCallPost()
        elif (parallelRegionEndMatch):
            self.processParallelRegionEndMatch(parallelRegionEndMatch)
            newState = "inside_subroutine_body"
        # elif (self.patterns.earlyReturnPattern.match(str(line))):
        #     raise UsageError("early return in the same subroutine within parallelRegion not allowed")
        elif (self.patterns.parallelRegionPattern.match(str(line))):
            raise UsageError("parallelRegion within parallelRegion not allowed")
        elif (self.patterns.subprocEndPattern.match(str(line))):
            raise UsageError("subprocedure end before @end parallelRegion")
        elif (self.patterns.subprocBeginPattern.match(str(line))):
            raise UsageError("subprocedure within subprocedure not allowed")
        elif templateMatch:
            raise UsageError("template directives are only allowed outside of subroutines")
        elif templateEndMatch:
            raise UsageError("template directives are only allowed outside of subroutines")
        else:
            self.processNoMatch()
        if newState == None:
            return
        if self.state == "inside_branch":
            self.stateBeforeBranch = newState
        else:
            self.state = newState

    def processInsideModuleDomainDependantRegionState(self, line):
        domainDependantEndMatch = self.patterns.domainDependantEndPattern.match(str(line))
        templateMatch = self.patterns.templatePattern.match(str(line))
        templateEndMatch = self.patterns.templateEndPattern.match(str(line))
        branchMatch = self.patterns.branchPattern.match(str(line))

        newState = None
        if branchMatch:
            self.processBranchMatch(branchMatch)
        elif domainDependantEndMatch:
            self.processDomainDependantEndMatch(domainDependantEndMatch)
            newState = "inside_module"
        elif (self.patterns.earlyReturnPattern.match(str(line))):
            raise UsageError("early return not allowed here")
        elif self.patterns.subprocCallPattern.match(str(line)):
            raise UsageError("subprocedure call within domainDependants not allowed")
        elif (self.patterns.parallelRegionEndPattern.match(str(line)) or self.patterns.parallelRegionPattern.match(str(line))):
            raise UsageError("parallelRegion within domainDependants not allowed")
        elif (self.patterns.subprocEndPattern.match(str(line))):
            raise UsageError("subprocedure end before @end domainDependant")
        elif (self.patterns.subprocBeginPattern.match(str(line))):
            raise UsageError("subprocedure within subprocedure not allowed")
        elif templateMatch:
            raise UsageError("template directives not allowed here")
        elif templateEndMatch:
            raise UsageError("template directives not allowed here")
        if newState == None:
            return
        if self.state == "inside_branch":
            self.stateBeforeBranch = newState
        else:
            self.state = newState

    def processInsideDomainDependantRegionState(self, line):
        domainDependantEndMatch = self.patterns.domainDependantEndPattern.match(str(line))
        templateMatch = self.patterns.templatePattern.match(str(line))
        templateEndMatch = self.patterns.templateEndPattern.match(str(line))
        branchMatch = self.patterns.branchPattern.match(str(line))

        newState = None
        if branchMatch:
            self.processBranchMatch(branchMatch)
        elif domainDependantEndMatch:
            self.processDomainDependantEndMatch(domainDependantEndMatch)
            newState = "inside_subroutine_body"
        elif (self.patterns.earlyReturnPattern.match(str(line))):
            raise UsageError("early return not allowed here")
        elif self.patterns.subprocCallPattern.match(str(line)):
            raise UsageError("subprocedure call within domainDependants not allowed")
        elif (self.patterns.parallelRegionEndPattern.match(str(line)) or self.patterns.parallelRegionPattern.match(str(line))):
            raise UsageError("parallelRegion within domainDependants not allowed")
        elif (self.patterns.subprocEndPattern.match(str(line))):
            raise UsageError("subprocedure end before @end domainDependant")
        elif (self.patterns.subprocBeginPattern.match(str(line))):
            raise UsageError("subprocedure within subprocedure not allowed")
        elif templateMatch:
            raise UsageError("template directives not allowed here")
        elif templateEndMatch:
            raise UsageError("template directives not allowed here")
        if newState == None:
            return
        if self.state == "inside_branch":
            self.stateBeforeBranch = newState
        else:
            self.state = newState

    def processUndefinedState(self, line):
        raise Exception("unexpected undefined parser state: %s" %(self.state))

    def processLine(self, line):
        global currLineNo
        currLineNo = self.lineNo
        self.currentLine = line

        #here we only load the current line into the branch analyzer for further use, we don't need the result of this method
        self.branchAnalyzer.currLevelAfterString(str(line))

        #analyse this line. handle the line according to current parser state.
        self.stateSwitch.get(self.state, self.processUndefinedState)(self.currentLine)

    def processFile(self, fileName):
        self.lineNo = 1
        self.fileName = fileName
        global currFile
        currFile = os.path.basename(fileName)
        for line in fileinput.input([fileName]):
            try:
                self.processLine(line)
            except UsageError as e:
                logging.error('Error: %s' %(str(e)), extra={"hfLineNo":currLineNo, "hfFile":currFile})
                sys.exit(1)
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                logging.error(
                    'Error when parsing file %s on line %i: %s; Print of line:%s\n' %(
                        str(fileName), self.lineNo, str(e), str(line).strip()
                    ),
                    extra={"hfLineNo":currLineNo, "hfFile":currFile}
                )
                logging.info(traceback.format_exc(), extra={"hfLineNo":currLineNo, "hfFile":currFile})
                sys.exit(1)
            self.lineNo += 1

        if (self.state != 'none'):
            logging.info(
                'Error when parsing file %s: File ended unexpectedly. Parser state: %s; Current Callee: %s; Current Subprocedure name: %s; Current Linenumber: %i; Current ArgumentParser: %s\n' %(
                    str(fileName), self.state, self.currCalleeName, self.currSubprocName, self.lineNo, str(self.currArgumentParser)
                ),
                extra={"hfLineNo":currLineNo, "hfFile":currFile}
            )
            sys.exit(1)
        del self.lineNo
        del self.fileName

class H90XMLCallGraphGenerator(H90CallGraphParser):
    doc = None
    routines = None
    modules = None
    templates = None
    calls = None
    currCallNode = None
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
        subProcName = subProcCallMatch.group(1)
        call = self.doc.createElement('call')
        call.setAttribute('caller', self.currSubprocName)
        call.setAttribute('callee', subProcName)
        if self.state == "inside_parallelRegion" or (self.state == "inside_branch" and self.stateBeforeBranch == "inside_parallelRegion"):
            call.setAttribute('parallelRegionPosition', 'surround')
        if (not firstDuplicateChild(self.calls, call)):
            self.calls.appendChild(call)
        self.currCallNode = call
        super(H90XMLCallGraphGenerator, self).processCallMatch(subProcCallMatch)

    def processArguments(self, nodeToAppendTo):
        arguments = self.doc.createElement('arguments')
        for symbolName in self.currArguments:
            argument = self.doc.createElement('argument')
            argument.setAttribute('symbolName', symbolName)
            arguments.appendChild(argument)
        nodeToAppendTo.appendChild(arguments)

    def processCallPost(self):
        self.processArguments(self.currCallNode)
        self.currCallNode = None
        super(H90XMLCallGraphGenerator, self).processCallPost()

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
        routine.setAttribute('module', self.currModuleName)
        self.routines.appendChild(routine)
        self.currSubprocNode = routine

    def processSubprocStartPost(self):
        self.processArguments(self.currSubprocNode)
        super(H90XMLCallGraphGenerator, self).processSubprocStartPost()

    def processParallelRegionMatch(self, parallelRegionMatch):
        super(H90XMLCallGraphGenerator, self).processParallelRegionMatch(parallelRegionMatch)
        self.currParallelRegionRelationNode, self.currParallelRegionTemplateNode = updateAndGetParallelRegionInfo(
            doc=self.doc,
            subroutineNode=self.currSubprocNode,
            parallelRegionSpecification=parallelRegionMatch.group(1),
            startLine=self.lineNo
        )

    def processParallelRegionEndMatch(self, parallelRegionEndMatch):
        super(H90XMLCallGraphGenerator, self).processParallelRegionEndMatch(parallelRegionEndMatch)
        self.currParallelRegionRelationNode.setAttribute("endLine", str(self.lineNo))
        self.currParallelRegionTemplateNode = None
        self.currParallelRegionRelationNode = None

    def processDomainDependantMatch(self, domainDependantMatch):
        super(H90XMLCallGraphGenerator, self).processDomainDependantMatch(domainDependantMatch)
        self.currDomainDependantRelationNode, _ = setTemplateInfos(
            self.doc,
            self.currModuleNode if self.state == 'inside_moduleDomainDependantRegion' or (self.state == "inside_branch" and self.stateBeforeBranch == "inside_moduleDomainDependantRegion") else self.currSubprocNode,
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
        if (self.state != 'inside_branch' and self.state != 'inside_moduleDomainDependantRegion') or (self.state == "inside_branch" and self.stateBeforeBranch != "inside_moduleDomainDependantRegion"):
            return
        addAndGetEntries(self.doc, self.currDomainDependantRelationNode, line)

    def processInsideDomainDependantRegionState(self, line):
        super(H90XMLCallGraphGenerator, self).processInsideDomainDependantRegionState(line)
        if (self.state != 'inside_branch' and self.state != 'inside_domainDependantRegion') or (self.state == "inside_branch" and self.stateBeforeBranch != "inside_domainDependantRegion"):
            return
        addAndGetEntries(self.doc, self.currDomainDependantRelationNode, line)

def getSymbolsByName(cgDoc, parentNode, parallelRegionTemplates=[], currentSymbolsByName={}, symbolAnalysisByRoutineNameAndSymbolName={}, isModuleSymbols=False):
    patterns = H90RegExPatterns.Instance()
    templatesAndEntries = getDomainDependantTemplatesAndEntries(cgDoc, parentNode)
    symbolsByName = {}
    parentName = parentNode.getAttribute('name')
    if parentName in [None, '']:
        raise Exception("parent node without identifier")
    for template, entry in templatesAndEntries:
        dependantName = entry.firstChild.nodeValue
        symbol = Symbol(dependantName, template, patterns)
        symbol.isModuleSymbol = isModuleSymbols
        analysis = symbolAnalysisByRoutineNameAndSymbolName.get(parentName, {}).get(dependantName)
        symbol.analysis = analysis
        symbol.loadDomainDependantEntryNodeAttributes(entry)
        if isModuleSymbols:
            symbol.loadModuleNodeAttributes(parentNode)
        else:
            symbol.loadRoutineNodeAttributes(parentNode, parallelRegionTemplates)

        existingSymbol = symbolsByName.get(dependantName)
        if existingSymbol == None:
            existingSymbol = currentSymbolsByName.get(dependantName)
        if existingSymbol != None:
            symbol.merge(existingSymbol)
        symbolsByName[dependantName] = symbol
    return symbolsByName

def getModuleNodesByName(cgDoc):
    moduleNodesByName = {}
    modules = cgDoc.getElementsByTagName('module')
    for module in modules:
        moduleName = module.getAttribute('name')
        if not moduleName or moduleName == '':
            raise Exception("Module without name.")
        moduleNodesByName[moduleName] = module
    return moduleNodesByName

def getParallelRegionData(cgDoc):
    parallelRegionTemplateRelationsByProcName = {}
    parallelRegionTemplatesByProcName = {}
    routineNodesByProcName = {}
    routineNodesByModule = {}
    regionsByID = regionTemplatesByID(cgDoc, 'parallelRegionTemplate')
    routines = cgDoc.getElementsByTagName('routine')
    for routine in routines:
        procName = routine.getAttribute('name')
        if procName in [None, '']:
            raise Exception("Procedure without name.")
        routineNodesByProcName[procName] = routine
        moduleName = routine.getAttribute('module')
        if moduleName not in [None, '']:
            routinesForModule = routineNodesByModule.get(moduleName, [])
            routinesForModule.append(routine)
            routineNodesByModule[moduleName] = routinesForModule
        regionTemplates = []
        parallelRegionsParents = routine.getElementsByTagName('activeParallelRegions')
        if parallelRegionsParents and len(parallelRegionsParents) > 0:
            templateRelations = parallelRegionsParents[0].getElementsByTagName('templateRelation')
            for templateRelation in templateRelations:
                idStr = templateRelation.getAttribute('id')
                if not idStr or idStr == '':
                    raise Exception("Template relation without id attribute.")
                regionTemplate = regionsByID.get(idStr, None)
                if not regionTemplate:
                    raise Exception("Template relation id %s could not be matched in procedure '%s'" %(idStr, procName))
                regionTemplates.append(regionTemplate)
            if len(templateRelations) > 0:
                parallelRegionTemplateRelationsByProcName[procName] = templateRelations
            if len(regionTemplates) > 0:
                parallelRegionTemplatesByProcName[procName] = regionTemplates
    return parallelRegionTemplateRelationsByProcName, parallelRegionTemplatesByProcName, routineNodesByProcName, routineNodesByModule

class H90CallGraphAndSymbolDeclarationsParser(H90CallGraphParser):
    cgDoc = None
    symbolsOnCurrentLine = None
    importsOnCurrentLine = None
    routineNodesByProcName = None
    moduleNodesByName = None
    routineNodesByModule = None
    parallelRegionTemplatesByProcName = None
    parallelRegionTemplateRelationsByProcName = None
    #$$$ remove this in case we never enable routine domain dependant specifications for module symbols (likely)
    # tentativeModuleSymbolsByName = None

    def __init__(self, cgDoc, moduleNodesByName=None, parallelRegionData=None):
        self.cgDoc = cgDoc
        self.symbolsOnCurrentLine = []
        self.importsOnCurrentLine = []
        #$$$ remove this in case we never enable routine domain dependant specifications for module symbols (likely)
        # self.tentativeModuleSymbolsByName = None
        if moduleNodesByName != None:
            self.moduleNodesByName = moduleNodesByName
        else:
            self.moduleNodesByName = getModuleNodesByName(cgDoc)

        if parallelRegionData == None:
            parallelRegionData = getParallelRegionData(cgDoc)
        self.parallelRegionTemplatesByProcName = parallelRegionData[1]
        self.parallelRegionTemplateRelationsByProcName = parallelRegionData[0]
        self.routineNodesByProcName = parallelRegionData[2]
        self.routineNodesByModule = parallelRegionData[3]
        super(H90CallGraphAndSymbolDeclarationsParser, self).__init__()

    def loadSymbolsFromTemplate(self, parentNode, parallelRegionTemplates, isModuleSymbols=False):
        self.currSymbolsByName.update(getSymbolsByName(
            self.cgDoc,
            parentNode,
            parallelRegionTemplates,
            currentSymbolsByName=self.currSymbolsByName,
            isModuleSymbols=isModuleSymbols,
            symbolAnalysisByRoutineNameAndSymbolName=self.symbolAnalysisByRoutineNameAndSymbolName \
                if hasattr(self, 'self.symbolAnalysisByRoutineNameAndSymbolName') \
                else {}
        ))
        logging.debug(
            "Symbols loaded from template. Symbols currently active in scope: %s. Module Symbol Property: %s" %(
                str(self.currSymbolsByName.values()),
                str([self.currSymbolsByName[symbolName].isModuleSymbol for symbolName in self.currSymbolsByName.keys()])
            ),
            extra={"hfLineNo":currLineNo, "hfFile":currFile}
        )

    def analyseSymbolInformationOnCurrentLine(self, line, analyseImports=True, isModuleSpecification=False):
        def loadAsAdditionalModuleSymbol(symbol):
            symbol.isModuleSymbol = True
            symbol.loadModuleNodeAttributes(self.moduleNodesByName[self.currModuleName])
            self.symbolsOnCurrentLine.append(symbol)
            self.currSymbolsByName[symbol.name] = symbol

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

        #$$$ remove this in case we never enable routine domain dependant specifications for module symbols (likely)
        # if isModuleSpecification:
        #     symbolNames = self.tentativeModuleSymbolsByName.keys() if self.tentativeModuleSymbolsByName else []
        #     for symbolName in symbolNames:
        #         symbol = self.tentativeModuleSymbolsByName[symbolName]
        #         declMatch = symbol.getDeclarationMatch(str(line))
        #         importMatch = None
        #         if analyseImports:
        #             importMatch = symbol.symbolImportPattern.match(str(line))
        #         if declMatch:
        #             self.symbolsOnCurrentLine.append(symbol)
        #             self.processSymbolDeclMatch(declMatch, symbol)
        #             loadAsAdditionalModuleSymbol(symbol)
        #         elif importMatch:
        #             self.importsOnCurrentLine.append(symbol)
        #             self.processSymbolImportMatch(importMatch, symbol)
        #             loadAsAdditionalModuleSymbol(symbol)

        #validate the symbols on the current declaration line: Do they match the requirements for Hybrid Fortran?
        lineDeclarationType = DeclarationType.UNDEFINED
        arrayDeclarationLine = None
        for symbol in self.symbolsOnCurrentLine:
            if symbol.isArray and arrayDeclarationLine == False or not symbol.isArray and arrayDeclarationLine == True:
                raise UsageError(
                    "Array symbols have been mixed with non-array symbols on the same line. This is invalid in Hybrid Fortran. Please move apart these declarations.\n"
                )
            arrayDeclarationLine = symbol.isArray

    def processSymbolDeclMatch(self, paramDeclMatch, symbol):
        '''process everything that happens per h90 declaration symbol'''
        symbol.isMatched = True
        symbol.loadDeclaration(
            paramDeclMatch,
            self.patterns,
            self.currArguments if isinstance(self.currArguments, list) else []
        )

    def processSymbolImportMatch(self, importMatch, symbol):
        symbol.isMatched = True
        moduleName = importMatch.group(1)
        moduleNode = self.moduleNodesByName.get(moduleName)
        symbol.loadImportInformation(importMatch, self.cgDoc, moduleNode)

    def processInsideModuleState(self, line):
        super(H90CallGraphAndSymbolDeclarationsParser, self).processInsideModuleState(line)
        self.analyseSymbolInformationOnCurrentLine(line, analyseImports=False, isModuleSpecification=True)

    def processInsideDeclarationsState(self, line):
        '''process everything that happens per h90 declaration line'''
        super(H90CallGraphAndSymbolDeclarationsParser, self).processInsideDeclarationsState(line)
        if (self.state != 'inside_branch' and self.state != 'inside_declarations') or (self.state == "inside_branch" and self.stateBeforeBranch != "inside_declarations"):
            return
        selectiveImportMatch = self.patterns.selectiveImportPattern.match(str(line))
        if selectiveImportMatch:
            self.processImplicitForeignModuleSymbolMatch(selectiveImportMatch)
        self.analyseSymbolInformationOnCurrentLine(line)

    def processImplicitForeignModuleSymbolMatch(self, importMatch):
        pass

    def processModuleBeginMatch(self, moduleBeginMatch):
        super(H90CallGraphAndSymbolDeclarationsParser, self).processModuleBeginMatch(moduleBeginMatch)
        moduleName = moduleBeginMatch.group(1)
        moduleNode = self.moduleNodesByName.get(moduleName)
        if not moduleNode:
            return
        self.loadSymbolsFromTemplate(moduleNode, None, isModuleSymbols=True)

        #$$$ remove this in case we never enable routine domain dependant specifications for module symbols (likely)
        #just in case a programmer (like myself when doing the original ASUCA HF physics when HF didn't yet have module capabilities) specified module symbols within routine nodes instead of the module :-S.
        # self.tentativeModuleSymbolsByName = {}
        # for routineNode in self.routineNodesByModule.get(moduleName, []):
        #     routineDependantsByName = getSymbolsByName(
        #         self.cgDoc,
        #         routineNode,
        #         currentSymbolsByName=self.currSymbolsByName
        #     )
        #     for dependantName in routineDependantsByName.keys():
        #         symbol = routineDependantsByName[dependantName]
        #         if symbol.isArgument:
        #             continue
        #         self.tentativeModuleSymbolsByName[dependantName] = symbol

    def processModuleEndMatch(self, moduleEndMatch):
        super(H90CallGraphAndSymbolDeclarationsParser, self).processProcEndMatch(moduleEndMatch)
        dependants = self.currSymbolsByName.keys()
        unmatched = []
        for dependant in dependants:
            if not self.currSymbolsByName[dependant].isModuleSymbol:
                raise UsageError("Dependant %s has been referenced in a domain dependant region inside a module, but has never been matched." %(dependant))
            if self.currSymbolsByName[dependant].isMatched:
                continue
            unmatched.append(dependant)
        if len(unmatched) != 0:
            raise UsageError("The following non-scalar domain dependant declarations could not be found within module %s: %s;\n\
                domains of first unmatched: %s"
                %(self.currModuleName, unmatched, str(self.currSymbolsByName[unmatched[0]].domains))
            )
        logging.debug("Clearing current symbol scope since the module definition is finished", extra={"hfLineNo":currLineNo, "hfFile":currFile})
        self.currSymbolsByName = {}
        #$$$ remove this in case we never enable routine domain dependant specifications for module symbols (likely)
        # self.tentativeModuleSymbolsByName = None

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
                #$$$ this code can probably be left away now that we analyze additional module symbols that haven't been declared domain dependant specifically within the module
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

    def processLine(self, line):
        super(H90CallGraphAndSymbolDeclarationsParser, self).processLine(line)
        self.symbolsOnCurrentLine = []
        self.importsOnCurrentLine = []

class H90XMLSymbolDeclarationExtractor(H90CallGraphAndSymbolDeclarationsParser):
    entryNodesBySymbolName = {}
    currSymbols = []
    symbolsByModuleNameAndSymbolName = None

    def __init__(self, cgDoc, symbolsByModuleNameAndSymbolName=None):
        super(H90XMLSymbolDeclarationExtractor, self).__init__(cgDoc)
        self.symbolsByModuleNameAndSymbolName = symbolsByModuleNameAndSymbolName

    def processSymbolAttributes(self, isModule=False):
        currSymbolNames = self.currSymbolsByName.keys()
        self.currSymbols = []
        if len(currSymbolNames) == 0:
            return
        self.currSymbols = [
            self.currSymbolsByName[symbolName]
            for symbolName in currSymbolNames
            if self.currSymbolsByName[symbolName].isModuleSymbol == isModule
        ]
        if len(self.currSymbols) == 0:
            return
        currParentName = self.currSubprocName if not isModule else self.currModuleName
        parentNode = self.routineNodesByProcName[currParentName] if not isModule else self.moduleNodesByName[currParentName]
        domainDependantRelationNodes = parentNode.getElementsByTagName("domainDependants")
        if domainDependantRelationNodes == None or len(domainDependantRelationNodes) == 0:
            raise Exception("we have active symbols (%s) loaded in %s but no domain dependant relation node can be found" %(
                self.currSymbols, currParentName
            ))
        domainDependantsRelationNode = domainDependantRelationNodes[0]
        domainDependantEntryNodes = domainDependantsRelationNode.getElementsByTagName("entry")
        if domainDependantEntryNodes == None or len(domainDependantEntryNodes) == 0:
            raise Exception("we have active symbols (%s) loaded in %s but no entry node can be found" %(
                self.currSymbols, currParentName
            ))
        self.entryNodesBySymbolName = {}
        for domainDependantEntryNode in domainDependantEntryNodes:
            self.entryNodesBySymbolName[domainDependantEntryNode.firstChild.nodeValue.strip()] = domainDependantEntryNode
        for symbol in self.currSymbols:
            entryNode = self.entryNodesBySymbolName.get(symbol.name)
            if entryNode:
                continue
            raise Exception("symbol %s is active but no information has been found in the codebase meta information" %(symbol))

    def storeCurrentSymbolAttributes(self, isModule=False):
        #store our symbol informations to the xml
        for symbol in self.currSymbols:
            if symbol.isModuleSymbol and isModule == False:
                continue
            entryNode = self.entryNodesBySymbolName.get(symbol.name)
            if not entryNode:
                continue
            symbol.storeDomainDependantEntryNodeAttributes(entryNode)

    def processImplicitForeignModuleSymbolMatch(self, importMatch):
        super(H90XMLSymbolDeclarationExtractor, self).processImplicitForeignModuleSymbolMatch(importMatch)
        if not self.symbolsByModuleNameAndSymbolName:
            return
        moduleName = importMatch.group(1)
        moduleSymbolsByName = self.symbolsByModuleNameAndSymbolName.get(moduleName)
        if not moduleSymbolsByName:
            return
        if moduleName == "":
            raise UsageError("import without module specified")
        symbolList = importMatch.group(2).split(',')
        for entry in symbolList:
            stripped = entry.strip()
            mappedImportMatch = self.patterns.singleMappedImportPattern.match(stripped)
            sourceSymbol = None
            symbolInScope = None
            if mappedImportMatch:
                symbolInScope = mappedImportMatch.group(1)
                sourceSymbol = mappedImportMatch.group(2)
            else:
                symbolInScope = stripped
                sourceSymbol = symbolInScope
            if moduleSymbolsByName.get(sourceSymbol) == None:
                continue
            relationNode, templateNode = setTemplateInfos(
                self.cgDoc,
                self.routineNodesByProcName.get(self.currSubprocName),
                specText="attribute(autoDom)",
                templateParentNodeName="domainDependantTemplates",
                templateNodeName="domainDependantTemplate",
                referenceParentNodeName="domainDependants"
            )
            addAndGetEntries(self.cgDoc, relationNode, symbolInScope)
            symbol = ImplicitForeignModuleSymbol(moduleName, symbolInScope, sourceSymbol, template=templateNode)
            symbol.isMatched = True
            self.currSymbolsByName[symbol.name] = symbol

    def processModuleEndMatch(self, moduleEndMatch):
        #get handles to currently active symbols -> temporarily save the handles
        self.processSymbolAttributes(isModule=True)
        logging.debug("exiting module %s. Storing informations for symbols %s" %(self.currModuleName, str(self.currSymbols)), extra={"hfLineNo":currLineNo, "hfFile":currFile})
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
        logging.debug("exiting procedure %s. Storing informations for symbols %s" %(self.currSubprocName, str(self.currSymbols)), extra={"hfLineNo":currLineNo, "hfFile":currFile})
        #finish parsing -> superclass destroys handles
        super(H90XMLSymbolDeclarationExtractor, self).processProcEndMatch(subProcEndMatch)
        #store our symbol informations to the xml
        self.storeCurrentSymbolAttributes()
        #throw away our handles
        self.entryNodesBySymbolName = {}
        self.currSymbols = []

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

def getSymbolsByModuleNameAndSymbolName(cgDoc, moduleNodesByName, symbolAnalysisByRoutineNameAndSymbolName={}):
    patterns = H90RegExPatterns.Instance()
    symbolsByModuleNameAndSymbolName = {}
    for moduleName in moduleNodesByName.keys():
        moduleNode = moduleNodesByName.get(moduleName)
        if not moduleNode:
            continue
        symbolsByModuleNameAndSymbolName[moduleName] = getSymbolsByName(
            cgDoc,
            moduleNode,
            isModuleSymbols=True,
            symbolAnalysisByRoutineNameAndSymbolName=symbolAnalysisByRoutineNameAndSymbolName
        )
        for symbolName in symbolsByModuleNameAndSymbolName[moduleName]:
            symbol = symbolsByModuleNameAndSymbolName[moduleName][symbolName]
            symbol.sourceModule = moduleName
    return symbolsByModuleNameAndSymbolName

def getSymbolsByRoutineNameAndSymbolName(cgDoc, routineNodesByProcName, parallelRegionTemplatesByProcName, symbolAnalysisByRoutineNameAndSymbolName={}):
    patterns = H90RegExPatterns.Instance()
    symbolsByRoutineNameAndSymbolName = {}
    for procName in routineNodesByProcName:
        routine = routineNodesByProcName[procName]
        procName = routine.getAttribute('name')
        symbolsByRoutineNameAndSymbolName[procName] = getSymbolsByName(
            cgDoc,
            routine,
            parallelRegionTemplatesByProcName.get(procName,[]),
            isModuleSymbols=False,
            symbolAnalysisByRoutineNameAndSymbolName=symbolAnalysisByRoutineNameAndSymbolName
        )
    return symbolsByRoutineNameAndSymbolName

class H90toF90Printer(H90CallGraphAndSymbolDeclarationsParser):
    currSubroutineImplementationNeedsToBeCommented = False
    currRoutineIsCallingParallelRegion = False
    currCalleeNode = None
    additionalParametersByKernelName = None
    additionalWrapperImportsByKernelName = None
    currAdditionalSubroutineParameters = None
    currAdditionalCompactedSubroutineParameters = None
    symbolsPassedInCurrentCallByName = None
    currParallelIterators = None
    intentPattern = None
    dimensionPattern = None
    implementationsByTemplateName = None
    codeSanitizer = None
    stateBeforeBranch = None
    currParallelRegionRelationNode = None
    currParallelRegionTemplateNode = None
    symbolsByModuleName = None
    symbolAnalysisByRoutineName = None
    symbolsByRoutineNameAndSymbolName = None
    currentLineNeedsPurge = False

    tab_insideSub = "\t\t"
    tab_outsideSub = "\t"

    def __init__(
        self,
        cgDoc,
        implementationsByTemplateName,
        outputStream=sys.stdout,
        moduleNodesByName=None,
        parallelRegionData=None,
        symbolAnalysisByRoutineNameAndSymbolName=None,
        symbolsByModuleNameAndSymbolName=None,
        symbolsByRoutineNameAndSymbolName=None
    ):
        super(H90toF90Printer, self).__init__(cgDoc, moduleNodesByName=moduleNodesByName, parallelRegionData=parallelRegionData)
        self.implementationsByTemplateName = implementationsByTemplateName
        self.outputStream = outputStream
        self.currRoutineIsCallingParallelRegion = False
        self.currSubroutineImplementationNeedsToBeCommented = False
        self.symbolsPassedInCurrentCallByName = {}
        self.additionalParametersByKernelName = {}
        self.additionalWrapperImportsByKernelName = {}
        self.currParallelIterators = []
        self.currentLine = ""
        self.currCalleeNode = None
        self.currAdditionalSubroutineParameters = []
        self.currAdditionalCompactedSubroutineParameters = []
        self.codeSanitizer = FortranCodeSanitizer()
        self.currParallelRegionRelationNode = None
        self.currParallelRegionTemplateNode = None
        try:
            if symbolAnalysisByRoutineNameAndSymbolName != None:
                self.symbolAnalysisByRoutineNameAndSymbolName = symbolAnalysisByRoutineNameAndSymbolName
            else:
                symbolAnalyzer = SymbolDependencyAnalyzer(self.cgDoc)
                self.symbolAnalysisByRoutineNameAndSymbolName = symbolAnalyzer.getSymbolAnalysisByRoutine()
            if symbolsByModuleNameAndSymbolName != None:
                self.symbolsByModuleNameAndSymbolName = symbolsByModuleNameAndSymbolName
            else:
                self.symbolsByModuleNameAndSymbolName = getSymbolsByModuleNameAndSymbolName(self.cgDoc, self.moduleNodesByName, self.symbolAnalysisByRoutineNameAndSymbolName)

            if symbolsByRoutineNameAndSymbolName != None:
                self.symbolsByRoutineNameAndSymbolName = symbolsByRoutineNameAndSymbolName
            else:
                self.symbolsByRoutineNameAndSymbolName = getSymbolsByRoutineNameAndSymbolName(
                    self.cgDoc,
                    self.routineNodesByProcName,
                    self.parallelRegionTemplatesByProcName,
                    self.symbolAnalysisByRoutineNameAndSymbolName
                )
        except UsageError as e:
            logging.error('Error: %s' %(str(e)), extra={"hfLineNo":currLineNo, "hfFile":currFile})
            sys.exit(1)
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            logging.critical('Error when initializing h90 conversion: %s' %(str(e)), extra={"hfLineNo":currLineNo, "hfFile":currFile})
            logging.info(traceback.format_exc())
            sys.exit(1)

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
            symbol_access_match = self.patterns.symbolAccessPattern.match(accessorString)
            if not symbol_access_match:
                return [], accessorString
            currBracketAnalyzer = BracketAnalyzer()
            return currBracketAnalyzer.getListOfArgumentsInOpenedBracketsAndRemainder(symbol_access_match.group(1))

        #match the symbol's postfix again in the current given line. (The prefix could have changed from the last match.)
        postfix = symbolMatch.group(3)
        postfixEscaped = re.escape(postfix)
        accessors, postfix = getAccessorsAndRemainder(postfix)

        if not self.implementation.supportsArbitraryDataAccessesOutsideOfKernels \
        and symbol.domains \
        and len(symbol.domains) > 0 \
        and not isInsideSubroutineCall \
        and not isPointerAssignment \
        and not symbol.isModuleSymbol \
        and not symbol.isHostSymbol \
        and self.state != "inside_parallelRegion" \
        and not (self.state == "inside_branch" and self.stateBeforeBranch == "inside_parallelRegion") \
        and self.routineNodesByProcName[self.currSubprocName].getAttribute("parallelRegionPosition") != "outside" \
        and len(accessors) != 0 \
        and ( \
            not self.implementation.supportsNativeMemsetsOutsideOfKernels \
            or any([accessor.strip() != ":" for accessor in accessors]) \
        ):
            logging.warning(
                "Dependant symbol %s accessed with accessor domains (%s) outside of a parallel region or subroutine call in subroutine %s(%s:%i)" %(
                    symbol.name,
                    str(accessors),
                    self.currSubprocName,
                    self.fileName,
                    self.lineNo
                ),
                extra={"hfLineNo":currLineNo, "hfFile":currFile}
            )

        #$$$ why are we checking for a present statement?
        accessPatternChangeRequired = False
        presentPattern = r"(.*?present\s*\(\s*)" + re.escape(symbol.nameInScope()) + postfixEscaped + r"\s*"
        currMatch = self.patterns.get(presentPattern).match(line)
        if not currMatch:
            pattern1 = r"(.*?(?:\W|^))" + re.escape(symbol.nameInScope()) + postfixEscaped + r"\s*"
            currMatch = self.patterns.get(pattern1).match(line)
            accessPatternChangeRequired = True
            if not currMatch:
                pattern2 = r"(.*?(?:\W|^))" + re.escape(symbol.name) + postfixEscaped + r"\s*"
                currMatch = self.patterns.get(pattern2).match(line)
                if not currMatch:
                    raise Exception(\
                        "Symbol %s is accessed in an unexpected way. Note: '_d' postfix is reserved for internal use. Cannot match one of the following patterns: \npattern1: '%s'\npattern2: '%s'" \
                        %(symbol.name, pattern1, pattern2))
        prefix = currMatch.group(1)
        numOfIndependentDomains = 0
        if accessPatternChangeRequired:
            numOfIndependentDomains = len(symbol.domains) - symbol.numOfParallelDomains
            offsets = []
            if len(accessors) != numOfIndependentDomains and len(accessors) != len(symbol.domains) and len(accessors) != 0:
                raise Exception("Unexpected array access for symbol %s (%s): Please use either %i (number of parallel independant dimensions) \
    or %i (dimensions of loaded domain for this array) or zero accessors. Symbol Domains: %s; Symbol Init Level: %i; Parallel Region Position: %s; Parallel Active: %s; Symbol template:\n%s\n" %(
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
            and not (self.state == "inside_branch" and self.stateBeforeBranch == "inside_parallelRegion") \
            and not isInsideSubroutineCall \
            and not isPointerAssignment \
            and not symbol.isModuleSymbol \
            and not symbol.isHostSymbol \
            and len(accessors) == 0 \
        ):
            symbol_access = symbol.nameInScope()
        else:
            symbol_access = symbol.accessRepresentation(
                iterators,
                offsets,
                self.currParallelRegionTemplateNode,
                inside_subroutine_call=isInsideSubroutineCall
            )
        logging.debug(
            "symbol %s on line %i rewritten to %s; change required: %s, accessors: %s, num of independent domains: %i" %(
                str(symbol),
                self.lineNo,
                symbol_access,
                accessPatternChangeRequired,
                str(accessors),
                numOfIndependentDomains
            ),
            extra={"hfLineNo":currLineNo, "hfFile":currFile}
        )
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
                    raise Exception("Symbol %s can't be matched again after adjustment. Adjusted portion: %s" %(symbol.name, processed))
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
        additionalImportsAndDeclarations = self.additionalParametersByKernelName.get(self.currCalleeName, ([],[]))
        additionalModuleSymbols = self.additionalWrapperImportsByKernelName.get(self.currCalleeName, [])
        for symbol in additionalImportsAndDeclarations[1] + additionalModuleSymbols:
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
        logging.debug(
            "In subroutine %s: Processing subroutine call to %s, parallel region position: %s" %(self.currSubprocName, self.currCalleeName, parallelRegionPosition),
            extra={"hfLineNo":currLineNo, "hfFile":currFile}
        )
        if self.currCalleeNode and parallelRegionPosition == "within":
            parallelRegionTemplates = self.parallelRegionTemplatesByProcName.get(self.currCalleeName)
            if parallelRegionTemplates == None or len(parallelRegionTemplates) == 0:
                raise Exception("No parallel region templates found for subroutine %s" %(self.currCalleeName))
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
        additionalSymbolsSeparated = self.additionalParametersByKernelName.get(self.currCalleeName)
        additionalImports, additionalDeclarations = self.additionalParametersByKernelName.get(self.currCalleeName, ([], []))
        toBeCompacted = []
        toBeCompacted, declarationPrefix, notToBeCompacted = self.listCompactedSymbolsAndDeclarationPrefixAndOtherSymbols(additionalImports + additionalDeclarations)
        compactedArrayList = []
        if len(toBeCompacted) > 0:
            compactedArrayName = "hfimp_%s" %(self.currCalleeName)
            compactedArray = FrameworkArray(compactedArrayName, declarationPrefix, domains=[("hfauto", str(len(toBeCompacted)))], isOnDevice=True)
            compactedArrayList = [compactedArray]
        additionalSymbols = sorted(notToBeCompacted + compactedArrayList)
        if len(additionalSymbols) > 0:
            adjustedLine = adjustedLine + "( &\n"
        else:
            adjustedLine = adjustedLine + "("
        symbolNum = 0
        bridgeStr = ", & !additional parameter inserted by framework\n" + self.tab_insideSub + "& "
        for symbol in additionalSymbols:
            hostName = symbol.nameInScope()
            adjustedLine = adjustedLine + hostName
            if symbolNum < len(additionalSymbols) - 1 or paramListMatch:
                adjustedLine = adjustedLine + bridgeStr
            symbolNum = symbolNum + 1
        if paramListMatch:
            adjustedLine = adjustedLine + self.processSymbolsAndGetAdjustedLine(paramListMatch.group(2), isInsideSubroutineCall=True)
        else:
            adjustedLine = adjustedLine + ")\n"

        callPreparationForSymbols = ""
        callPostForSymbols = ""
        # if self.currCalleeNode and self.currCalleeNode.getAttribute("parallelRegionPosition") == "within":
        if self.currCalleeNode:
            if self.state != "inside_subroutine_call" and not (self.state == "inside_branch" and self.stateBeforeBranch == "inside_subroutine_call"):
                currSubprocNode = self.routineNodesByProcName.get(self.currSubprocName)
                callPreparationForSymbols = ""
                callPostForSymbols = ""
                for symbol in self.symbolsPassedInCurrentCallByName.values():
                    if symbol.isHostSymbol:
                        continue
                    symbolsInCalleeByName = self.symbolsByRoutineNameAndSymbolName.get(self.currCalleeName)
                    if symbolsInCalleeByName == None:
                        raise Exception("No collection of symbols found for callee %s" %(self.currCalleeName))
                    symbolNameInCallee = None
                    for symbolName in symbolsInCalleeByName:
                        symbolAnalysisPerCallee = self.symbolAnalysisByRoutineNameAndSymbolName.get(self.currCalleeName, {}).get(symbolName)
                        if symbolAnalysisPerCallee == None or len(symbolAnalysisPerCallee) == 0:
                            continue
                        if symbolAnalysisPerCallee[0].aliasNamesByRoutineName.get(self.currSubprocName) == symbol.name:
                            symbolNameInCallee = symbolName
                            break
                    if symbolNameInCallee == None:
                        continue #this symbol isn't passed in to the callee
                    symbolsInCalleeByName = self.symbolsByRoutineNameAndSymbolName.get(self.currCalleeName)
                    if symbolsInCalleeByName == None:
                        raise Exception("No collection of symbols found for callee %s" %(self.currCalleeName))
                    symbolInCallee = symbolsInCalleeByName.get(symbolNameInCallee)
                    if symbolInCallee == None:
                        raise Exception("Symbol %s's data expected for callee %s, but could not be found" %(
                            symbolNameInCallee,
                            self.currCalleeName
                        ))
                    callPreparationForSymbols += self.implementation.callPreparationForPassedSymbol(
                        currSubprocNode,
                        symbolInCaller=symbol,
                        symbolInCallee=symbolInCallee
                    )
                    callPostForSymbols += self.implementation.callPostForPassedSymbol(
                        currSubprocNode,
                        symbolInCaller=symbol,
                        symbolInCallee=symbolInCallee
                    )
                adjustedLine = self.processCallPostAndGetAdjustedLine(adjustedLine)

        if self.state != "inside_subroutine_call" and not (self.state == "inside_branch" and self.stateBeforeBranch == "inside_subroutine_call"):
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
            adjustedLine = adjustedLine + symbol.nameInScope()
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
        branchSettingText = branchMatch.group(1).strip()
        branchSettings = branchSettingText.split(",")
        if len(branchSettings) != 1:
            raise Exception("Invalid number of branch settings.")
        branchSettingMatch = re.match(r'(\w*)\s*\(\s*(\w*)\s*\)', branchSettings[0].strip(), re.IGNORECASE)
        if not branchSettingMatch:
            raise Exception("Invalid branch setting definition.")
        if self.state == "inside_branch":
            raise Exception("Nested @if branches are not allowed in Hybrid Fortran")

        self.stateBeforeBranch = self.state
        if branchSettingMatch.group(1) == "parallelRegion":
            if branchSettingMatch.group(2) == self.routineNodesByProcName[self.currSubprocName].getAttribute('parallelRegionPosition').strip():
                self.state = 'inside_branch'
            else:
                self.state = 'inside_ignore'
        elif branchSettingMatch.group(1) == "architecture":
            if branchSettingMatch.group(2) == self.implementation.architecture:
                self.state = 'inside_branch'
            else:
                self.state = 'inside_ignore'
        else:
            raise Exception("Invalid branch setting definition: Currently only parallelRegion and architecture setting accepted.")
        self.prepareLine("","")
        self.currentLineNeedsPurge = True

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
        additionalImportsForOurSelves, additionalDeclarationsForOurselves = self.implementation.getAdditionalKernelParameters(
            self.cgDoc,
            routineNode,
            self.moduleNodesByName[self.currModuleName],
            self.parallelRegionTemplatesByProcName.get(subprocName),
            self.currSymbolsByName
        )
        toBeCompacted, declarationPrefix, otherImports = self.listCompactedSymbolsAndDeclarationPrefixAndOtherSymbols(
            additionalImportsForOurSelves + additionalDeclarationsForOurselves
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
            raise Exception("Caller library not found.")
        calls = callsLibraries[0].getElementsByTagName("call")
        for call in calls:
            if call.getAttribute("caller") != self.currSubprocName:
                continue
            calleeName = call.getAttribute("callee")
            callee = self.routineNodesByProcName.get(calleeName)
            if not callee:
                continue
            additionalImportsForDeviceCompatibility, additionalDeclarationsForDeviceCompatibility = self.implementation.getAdditionalKernelParameters(
                self.cgDoc,
                callee,
                self.moduleNodesByName[self.currModuleName],
                self.parallelRegionTemplatesByProcName.get(calleeName),
                self.currSymbolsByName
            )
            for symbol in additionalImportsForDeviceCompatibility + additionalDeclarationsForDeviceCompatibility:
                symbol.resetScope()
                symbol.nameOfScope = self.currSubprocName
            if 'DEBUG_PRINT' in self.implementation.optionFlags:
                tentativeAdditionalImports = getModuleArraysForCallee(
                    calleeName,
                    self.symbolAnalysisByRoutineNameAndSymbolName,
                    self.symbolsByModuleNameAndSymbolName
                )
                additionalImports = [
                    symbol for symbol in tentativeAdditionalImports
                    if symbol.name not in self.symbolsByRoutineNameAndSymbolName.get(self.currSubprocName, {}) and \
                    symbol.analysis.argumentIndexByRoutineName.get(subprocName, -1) == -1
                ]
                additionalImportsByName = {}
                for symbol in additionalImports:
                    additionalImportsByName[symbol.name] = symbol
                    symbol.resetScope()
                    symbol.nameOfScope = self.currSubprocName
                self.additionalWrapperImportsByKernelName[calleeName] = additionalImportsByName.values()
            self.additionalParametersByKernelName[calleeName] = (additionalImportsForDeviceCompatibility, additionalDeclarationsForDeviceCompatibility)
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
        self.additionalParametersByKernelName = {}
        self.additionalWrapperImportsByKernelName = {}
        self.currAdditionalSubroutineParameters = []
        self.currAdditionalCompactedSubroutineParameters = []
        self.currSubroutineImplementationNeedsToBeCommented = False
        super(H90toF90Printer, self).processProcEndMatch(subProcEndMatch)

    def processParallelRegionMatch(self, parallelRegionMatch):
        super(H90toF90Printer, self).processParallelRegionMatch(parallelRegionMatch)
        logging.debug(
            "...parallel region starts on line %i with active symbols %s" %(self.lineNo, str(self.currSymbolsByName.values())),
            extra={"hfLineNo":currLineNo, "hfFile":currFile}
        )
        self.prepareActiveParallelRegion('parallelRegionBegin')

    def processParallelRegionEndMatch(self, parallelRegionEndMatch):
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

    def listCompactedSymbolsAndDeclarationPrefixAndOtherSymbols(self, additionalImports):
        toBeCompacted = []
        otherImports = []
        declarationPrefix = None
        for symbol in additionalImports:
            declType = symbol.declarationType

            # compact the imports of real type. Background: Experience has shown that too many
            # symbols passed to kernels, such that parameter list > 256Byte, can cause strange behavior. (corruption
            # of parameter list leading to launch failures)
            #-> need to pack all real symbols into an array to make it work reliably for cases where many reals are imported
            # why not integers? because they can be used as array boundaries.
            # Note: currently only a single real type per subroutine is supported for compaction
            if declType in [DeclarationType.FOREIGN_MODULE_SCALAR, DeclarationType.LOCAL_MODULE_SCALAR] \
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

        if self.state != "inside_declarations" and self.state != "inside_module" and self.state != "inside_subroutine_call" \
        and not (self.state in ["inside_branch", "inside_ignore"] and self.stateBeforeBranch in ["inside_declarations", "inside_module", "inside_subroutine_call"]):
            additionalDeclarationsStr = ""

            #TODO $$$: most of the following code should probably be handled within implementation classes

            #########################################################################
            # gather additional symbols for ourselves                               #
            #########################################################################
            ourSymbolsToAdd = sorted(
                self.currAdditionalSubroutineParameters + self.currAdditionalCompactedSubroutineParameters
            )
            additionalImports = []
            if 'DEBUG_PRINT' in self.implementation.optionFlags:
                callsLibraries = self.cgDoc.getElementsByTagName("calls")
                calls = callsLibraries[0].getElementsByTagName("call")
                for call in calls:
                    if call.getAttribute("caller") != self.currSubprocName:
                        continue
                    calleeName = call.getAttribute("callee")
                    additionalImports += self.additionalWrapperImportsByKernelName.get(calleeName, [])

            #########################################################################
            # gather symbols to be packed for called kernels                        #
            #########################################################################
            packedRealSymbolsByCalleeName = {}
            compactionDeclarationPrefixByCalleeName = {}
            for calleeName in self.additionalParametersByKernelName.keys():
                additionalImports, additionalDeclarations = self.additionalParametersByKernelName[calleeName]
                toBeCompacted, declarationPrefix, _ = self.listCompactedSymbolsAndDeclarationPrefixAndOtherSymbols(
                    additionalImports + additionalDeclarations
                )
                if len(toBeCompacted) > 0:
                    compactionDeclarationPrefixByCalleeName[calleeName] = declarationPrefix
                    packedRealSymbolsByCalleeName[calleeName] = toBeCompacted

            #########################################################################
            # mark in code, include additional symbols for kernel calls             #
            #########################################################################
            numberOfAdditionalDeclarations = len(self.additionalParametersByKernelName.keys()) \
                + len(ourSymbolsToAdd) \
                + len(additionalImports) \
                + len(packedRealSymbolsByCalleeName.keys())
            if numberOfAdditionalDeclarations > 0:
                additionalDeclarationsStr = "\n" + self.tab_insideSub + \
                 "! ****** additional symbols inserted by framework to emulate device support of language features\n"

            #########################################################################
            # create declaration lines for symbols for ourself                      #
            #########################################################################
            for symbol in ourSymbolsToAdd:
                purgeList=['public']
                if symbol.isCompacted:
                    purgeList=['intent', 'public']
                additionalDeclarationsStr += self.tab_insideSub + self.implementation.adjustDeclarationForDevice(
                    self.tab_insideSub +
                        symbol.getDeclarationLineForAutomaticSymbol(purgeList).strip(),
                    [symbol],
                    self.currRoutineIsCallingParallelRegion,
                    self.routineNodesByProcName[self.currSubprocName].getAttribute('parallelRegionPosition')
                ).rstrip() + " ! type %i symbol added for this subroutine\n" %(symbol.declarationType)
                logging.debug(
                    "...In subroutine %s: Symbol %s additionally declared" %(self.currSubprocName, symbol),
                    extra={"hfLineNo":currLineNo, "hfFile":currFile}
                )

            #########################################################################
            # create declaration lines for called kernels                           #
            #########################################################################
            for calleeName in self.additionalParametersByKernelName.keys():
                additionalImports, additionalDeclarations = self.additionalParametersByKernelName[calleeName]
                additionalImportSymbolsByName = {}
                for symbol in additionalImports:
                    additionalImportSymbolsByName[symbol.name] = symbol

                for symbol in additionalDeclarations:
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

                    additionalDeclarationsStr += self.implementation.adjustDeclarationForDevice(
                        symbol.getDeclarationLineForAutomaticSymbol(purgeList=['intent', 'public']).strip(),
                        [symbol],
                        self.currRoutineIsCallingParallelRegion,
                        self.routineNodesByProcName[self.currSubprocName].getAttribute('parallelRegionPosition')
                    ).rstrip() + " ! type %i symbol added for callee %s\n" %(symbol.declarationType, calleeName)
                    logging.debug(
                        "...In subroutine %s: Symbol %s additionally declared and passed to %s" %(self.currSubprocName, symbol, calleeName),
                        extra={"hfLineNo":currLineNo, "hfFile":currFile}
                    )
                #TODO: move this into implementation classes
                toBeCompacted = packedRealSymbolsByCalleeName.get(calleeName, [])
                if len(toBeCompacted) > 0:
                    #TODO: generalize for cases where we don't want this to be on the device (e.g. put this into Implementation class)
                    compactedArrayName = "hfimp_%s" %(calleeName)
                    compactedArray = FrameworkArray(
                        compactedArrayName,
                        compactionDeclarationPrefixByCalleeName[calleeName],
                        domains=[("hfauto", str(len(toBeCompacted)))],
                        isOnDevice=True
                    )
                    additionalDeclarationsStr += self.implementation.adjustDeclarationForDevice(
                        compactedArray.getDeclarationLineForAutomaticSymbol().strip(),
                        [compactedArray],
                        self.currRoutineIsCallingParallelRegion,
                        self.routineNodesByProcName[self.currSubprocName].getAttribute('parallelRegionPosition')
                    ).rstrip() + " ! compaction array added for callee %s\n" %(calleeName)
                    logging.debug(
                        "...In subroutine %s: Symbols %s packed into array %s" %(self.currSubprocName, toBeCompacted, compactedArrayName),
                        extra={"hfLineNo":currLineNo, "hfFile":currFile}
                    )

            additionalDeclarationsStr += self.implementation.declarationEnd( \
                self.currSymbolsByName.values() + additionalImports, \
                self.currRoutineIsCallingParallelRegion, \
                self.routineNodesByProcName[self.currSubprocName], \
                self.parallelRegionTemplatesByProcName.get(self.currSubprocName)
            )

            #########################################################################
            # additional symbols for kernels to be packed                           #
            #########################################################################
            calleesWithPackedReals = packedRealSymbolsByCalleeName.keys()
            for calleeName in calleesWithPackedReals:
                for idx, symbol in enumerate(sorted(packedRealSymbolsByCalleeName[calleeName])):
                    #$$$ clean this up, the hf_imp prefix should be decided within the symbol class
                    additionalDeclarationsStr += "hfimp_%s(%i) = %s" %(calleeName, idx+1, symbol.nameInScope()) + \
                         " ! type %i symbol compaction for callee %s\n" %(symbol.declarationType, calleeName)

            #########################################################################
            # additional symbols for ourselves to be unpacked                       #
            #########################################################################
            #TODO: move this into implementation classes
            for idx, symbol in enumerate(self.currAdditionalCompactedSubroutineParameters):
                #$$$ clean this up, the hf_imp prefix should be decided within the symbol class
                additionalDeclarationsStr += "%s = hfimp_%s(%i)" %(symbol.nameInScope(), self.currSubprocName, idx+1) + \
                         " ! additional type %i symbol compaction\n" %(symbol.declarationType)

            #########################################################################
            # mark the end of additional includes in code                           #
            #########################################################################
            if numberOfAdditionalDeclarations > 0:
                additionalDeclarationsStr += "! ****** end additional symbols\n\n"

            self.prepareLine(additionalDeclarationsStr, self.tab_insideSub)

        if self.state != "inside_declarations" and not (self.state == "inside_branch" and self.stateBeforeBranch == "inside_declarations"):
            return


        baseline = line
        if self.currentLineNeedsPurge:
            baseline = ""
        adjustedLine = baseline

        for symbol in self.symbolsOnCurrentLine:
            match = symbol.getDeclarationMatch(str(adjustedLine))
            if not match:
                raise Exception("Symbol %s not found on a line where it has already been identified before. Current string to search: %s" \
                    %(symbol, adjustedLine))
            adjustedLine = symbol.getAdjustedDeclarationLine(match, \
                self.parallelRegionTemplatesByProcName.get(self.currSubprocName), \
                self.patterns.dimensionPattern \
            )

        if adjustedLine != baseline:
            #$$$ this is scary. isn't there a better state test for this?
            adjustedLine = purgeDimensionAndGetAdjustedLine(adjustedLine, self.patterns)
            adjustedLine = str(adjustedLine).rstrip() + "\n"

        if len(self.symbolsOnCurrentLine) > 0:
            adjustedLine = self.implementation.adjustDeclarationForDevice(adjustedLine, \
                self.symbolsOnCurrentLine, \
                self.currRoutineIsCallingParallelRegion, \
                self.routineNodesByProcName[self.currSubprocName].getAttribute('parallelRegionPosition') \
            )

        for symbol in self.importsOnCurrentLine:
            match = symbol.symbolImportPattern.match(str(adjustedLine))
            if not match:
                continue #$$$ when does this happen?
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
            if self.state != 'inside_subroutine_call' and not (self.state == "inside_branch" and self.stateBeforeBranch == "inside_subroutine_call"):
                self.processCallPost()
            return

        subProcEndMatch = self.patterns.subprocEndPattern.match(str(line))
        if (subProcEndMatch):
            self.processProcEndMatch(subProcEndMatch)
            if self.state == "inside_branch":
                self.stateBeforeBranch = "inside_module"
            else:
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
                raise Exception("No parallel region template relation found for this region.")
            for templateRelation in templateRelations:
                startLine = templateRelation.getAttribute("startLine")
                if startLine in [None, '']:
                    continue
                startLineInt = 0
                try:
                    startLineInt = int(startLine)
                except ValueError:
                    raise Exception("Invalid startLine definition for parallel region template relation: %s\n. All active template relations: %s\nRoutine node: %s" %(
                        templateRelation.toxml(),
                        [templateRelation.toxml() for templateRelation in templateRelations],
                        self.routineNodesByProcName[self.currSubprocName].toprettyxml()
                    ))
                if startLineInt == self.lineNo:
                    self.currParallelRegionRelationNode = templateRelation
                    break
            else:
                raise Exception("No parallel region template relation was matched for the current linenumber.")
            logging.debug(
                "parallel region detected on line %i with template relation %s" %(self.lineNo, self.currParallelRegionRelationNode.toxml()),
                extra={"hfLineNo":currLineNo, "hfFile":currFile}
            )
            templates = self.parallelRegionTemplatesByProcName.get(self.currSubprocName)
            if templates == None or len(templates) == 0:
                raise Exception("No parallel region template found for this region.")
            activeTemplateID = self.currParallelRegionRelationNode.getAttribute("id")
            for template in templates:
                if template.getAttribute("id") == activeTemplateID:
                    self.currParallelRegionTemplateNode = template
                    break
            else:
                raise Exception("No parallel region template has matched the active template ID.")
            self.currParallelIterators = self.implementation.getIterators(self.currParallelRegionTemplateNode)
            if len(self.currParallelIterators) > 0:
                self.processParallelRegionMatch(parallelRegionMatch)
                if self.state == "inside_branch":
                    self.stateBeforeBranch = "inside_parallelRegion"
                else:
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
            if self.state == "inside_branch":
                self.stateBeforeBranch = "inside_domainDependantRegion"
            else:
                self.state = 'inside_domainDependantRegion'
            self.processDomainDependantMatch(domainDependantMatch)
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
                message = self.implementation.warningOnUnrecognizedSubroutineCallInParallelRegion(
                    self.currSubprocName,
                    subProcCallMatch.group(1)
                )
                if message != "":
                    logging.warning(message, extra={"hfLineNo":currLineNo, "hfFile":currFile})
            self.processCallMatch(subProcCallMatch)
            if self.state != 'inside_subroutine_call' and not (self.state == "inside_branch" and self.stateBeforeBranch == "inside_subroutine_call"):
                self.processCallPost()
            return

        parallelRegionEndMatch = self.patterns.parallelRegionEndPattern.match(str(line))
        if (parallelRegionEndMatch):
            self.processParallelRegionEndMatch(parallelRegionEndMatch)
            self.state = "inside_subroutine_body"
            if self.state == "inside_branch":
                self.stateBeforeBranch = "inside_subroutine_body"
            else:
                self.state = 'inside_subroutine_body'
            return

        if (self.patterns.parallelRegionPattern.match(str(line))):
            raise Exception("parallelRegion within parallelRegion not allowed")
        if (self.patterns.subprocEndPattern.match(str(line))):
            raise Exception("subprocedure end before @end parallelRegion")
        if (self.patterns.subprocBeginPattern.match(str(line))):
            raise Exception("subprocedure within subprocedure not allowed")

        adjustedLine = ""
        whileLoopMatch = self.patterns.whileLoopPattern.match(str(line))
        loopMatch = self.patterns.loopPattern.match(str(line))
        if whileLoopMatch == None and loopMatch != None:
            adjustedLine += self.implementation.loopPreparation().strip() + '\n'
        adjustedLine += self.processSymbolsAndGetAdjustedLine(line, isInsideSubroutineCall=False)
        self.prepareLine(adjustedLine, self.tab_insideSub)

    def processInsideDomainDependantRegionState(self, line):
        super(H90toF90Printer, self).processInsideDomainDependantRegionState(line)
        self.prepareLine("", "")

    def processInsideModuleDomainDependantRegionState(self, line):
        super(H90toF90Printer, self).processInsideModuleDomainDependantRegionState(line)
        self.prepareLine("", "")

    def processSpecificationBeginning(self):
        adjustedLine = self.currentLine
        additionalImports = sum(
            [self.additionalParametersByKernelName[kernelName][0] for kernelName in self.additionalParametersByKernelName.keys()],
            []
        ) + sum(
            [self.additionalWrapperImportsByKernelName[kernelName] for kernelName in self.additionalWrapperImportsByKernelName.keys()],
            []
        )
        logging.debug(
            "curr Module: %s; additional imports: %s" %(
                self.currModuleName,
                ["%s: %s from %s" %(symbol.name, symbol.declarationType, symbol.sourceModule) for symbol in additionalImports]
            ),
            extra={"hfLineNo":currLineNo, "hfFile":currFile}
        )
        for symbol in additionalImports:
            if symbol.declarationType not in [DeclarationType.FOREIGN_MODULE_SCALAR, DeclarationType.LOCAL_ARRAY, DeclarationType.MODULE_ARRAY]:
                continue
            adjustedLine = adjustedLine + "use %s, only : %s => %s\n" %(
                symbol.sourceModule,
                symbol.nameInScope(),
                symbol.sourceSymbol if symbol.sourceSymbol not in [None, ""] else symbol.name
            )

        self.prepareLine(adjustedLine + self.implementation.additionalIncludes(), self.tab_insideSub)

    def processInsideBranch(self, line):
        if self.patterns.branchEndPattern.match(str(line)):
            self.state = self.stateBeforeBranch
            self.stateBeforeBranch = None
        else:
            self.stateSwitch.get(self.stateBeforeBranch, self.processUndefinedState)(line)
        if self.state != "inside_branch":
            self.prepareLine("", "")
            return

    def processInsideIgnore(self, line):
        if self.patterns.branchEndPattern.match(str(line)):
            self.state = self.stateBeforeBranch
            self.stateBeforeBranch = None
        self.prepareLine("", "")

    def processLine(self, line):
        self.currentLineNeedsPurge = False
        super(H90toF90Printer, self).processLine(line)
        self.outputStream.write(self.currentLine)

    def processFile(self, fileName):
        self.outputStream.write(self.implementation.filePreparation(fileName))
        super(H90toF90Printer, self).processFile(fileName)

    #TODO: remove tab argument everywhere
    def prepareLine(self, line, tab):
        self.currentLine = self.codeSanitizer.sanitizeLines(line)
        logging.debug(
            "[%s]:%i:%s" %(self.state,self.lineNo,self.currentLine),
            extra={"hfLineNo":currLineNo, "hfFile":currFile}
        )
