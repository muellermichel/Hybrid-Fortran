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

import re, logging
from tools.commons import BracketAnalyzer, Singleton, UsageError, findRightMostOccurrenceNotInsideQuotes, \
    splitIntoComponentsAndRemainder, getComponentNameAndBracketContent
from tools.patterns import RegExPatterns

def purgeDimensionAndGetAdjustedLine(line):
    match = RegExPatterns.Instance().dimensionPattern.match(line)
    if not match:
        return line
    else:
        return match.group(1) + match.group(3)

def getAccessorsAndRemainder(accessorString):
    symbolAccessString_match = RegExPatterns.Instance().symbolAccessPattern.match(accessorString)
    if not symbolAccessString_match:
        return [], accessorString
    currBracketAnalyzer = BracketAnalyzer()
    return currBracketAnalyzer.getListOfArgumentsInOpenedBracketsAndRemainder(symbolAccessString_match.group(1))

def parseSpecification(line):
    def parseDataObjectsAndRemainder(specRightHandSide):
        dataObjects, remainder = splitIntoComponentsAndRemainder(specRightHandSide)
        if len(dataObjects) == 0:
            raise UsageError("no data objects defined on this line")
        if len(dataObjects) > 1 and remainder.strip() != "":
            raise UsageError("invalid right hand side specification")
        parsedDataObjects = []
        for dataObject in dataObjects:
            parsedDataObjects.append(getComponentNameAndBracketContent(dataObject))
        return tuple(parsedDataObjects), remainder

    patterns = RegExPatterns.Instance()
    multiSpecMatch = patterns.multiSpecPattern.match(line)
    if multiSpecMatch:
        declarationComponents, remainder = splitIntoComponentsAndRemainder(multiSpecMatch.group(1))
        if len(declarationComponents) == 0 or not patterns.standardTypePattern.match(declarationComponents[0]):
            return None, None, None
        parsedDataObjects, _ = parseDataObjectsAndRemainder(multiSpecMatch.group(2))
        return ", ".join(declarationComponents), parsedDataObjects, ""
    declarationComponents, remainder = splitIntoComponentsAndRemainder(line)
    if remainder.strip() == "" or len(declarationComponents) == 0:
        return None, None, None
    if len(declarationComponents) == 0 or not patterns.standardTypePattern.match(declarationComponents[0]):
        return None, None, None
    parsedDataObjects, remainder = parseDataObjectsAndRemainder(remainder)
    if len(parsedDataObjects) != 1:
        raise UsageError("invalid number of data objects specified on this declaration line")
    return ", ".join(declarationComponents), parsedDataObjects, remainder

def getSymbolAccessStringAndRemainder(
    symbol,
    parallelIterators,
    parallelRegionTemplate,
    accessorString,
    callee=None,
    isPointerAssignment=False
):
    accessors = []
    remainder = accessorString
    if len(symbol.domains) > 0: #0 domains could be an external function - need to retain postfix
        accessors, remainder = getAccessorsAndRemainder(accessorString)
    symbolAccessString = symbol.accessRepresentation(
        parallelIterators,
        accessors,
        parallelRegionTemplate,
        isPointerAssignment=isPointerAssignment,
        isInsideParallelRegion=parallelRegionTemplate != None,
        callee=callee
    )
    return symbolAccessString, remainder

@Singleton
class ConversionOptions:
    debugPrint = False

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
    def __init__(self):
        self.tabIncreasingPattern = re.compile(r'\s*(?:(?:module|select|do|subroutine|function|program|attributes)|if\W.*?\Wthen)(?:\W|$).*', re.IGNORECASE)
        self.tabDecreasingPattern = re.compile(r'\s*end\s*(?:module|select|do|subroutine|function|program|if)(?:\W|$).*', re.IGNORECASE)
        self.commentedPattern = re.compile(r'\s*\!', re.IGNORECASE)
        self.openMPLinePattern = re.compile(r'\s*\!\$OMP.*', re.IGNORECASE)
        self.openACCLinePattern = re.compile(r'\s*\!\$ACC.*', re.IGNORECASE)
        self.preprocessorPattern = re.compile(r'\s*\#', re.IGNORECASE)
        self.currNumOfTabs = 0
        self.emptyLinesInARow = 0

    def sanitizeLines(self, line, toBeCommented=False, howManyCharsPerLine=132, commentChar="!"):
        strippedRawLine = line.strip()
        if strippedRawLine == "" and self.emptyLinesInARow > 1:
            return ""
        if strippedRawLine == "":
            self.emptyLinesInARow += 1
            return "\n"

        self.emptyLinesInARow = 0
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
            if strippedRawLine == "" and self.emptyLinesInARow > 1:
                continue
            if strippedLine == "":
                self.emptyLinesInARow += 1
                tabbedCodeLines.append("")
                continue
            self.emptyLinesInARow = 0
            if self.commentedPattern.match(strippedLine):
                tabbedCodeLines.append(strippedLine)
            elif self.preprocessorPattern.match(strippedLine):
                #note: ifort's preprocessor can't handle preprocessor lines with leading whitespace -.-
                #=> catch this case and strip any whitespace.
                tabbedCodeLines.append(strippedLine)
            elif self.tabDecreasingPattern.match(strippedLine):
                self.currNumOfTabs = max(0, self.currNumOfTabs - 1)
                tabbedCodeLines.append(self.currNumOfTabs * "\t" + strippedLine)
            elif self.tabIncreasingPattern.match(strippedLine):
                tabbedCodeLines.append(self.currNumOfTabs * "\t" + strippedLine)
                self.currNumOfTabs = min(5, self.currNumOfTabs + 1)
            else:
                tabbedCodeLines.append(self.currNumOfTabs * "\t" + strippedLine)

        return "\n".join(tabbedCodeLines) + "\n"