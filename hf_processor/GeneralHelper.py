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
#  Procedure        FileHelper.py                                      #
#  Date             2012/07/27                                         #
#  Author           Michel Müller (AOKI Laboratory)                    #
#**********************************************************************#


import os
import sys
import re

def stripWhitespace(inputStr):
    match = re.match(r'\s*(.*)\s*', inputStr)
    if not match:
        raise Exception("Unexpected error: Whitespace could not be removed from string %s" %(inputStr))
    return match.group(1)

def openFile(file_name, mode):
    """Open a file."""
    try:
        the_file = open(file_name, mode)
    except(IOError), e:
        sys.stderr.write("Unable to open the file %s. Ending program. Error: %s\n" %(file_name, e))
        sys.exit(1)
    else:
        return the_file

def enum(*sequential, **named):
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)

def areIndexesWithinQuotes(stringToSearch):
    #build up a colored list, showing whether an index inside stringToSearch is in quotes or not.
    #nested quotes such as "hello'world' foobar" are not supported!
    quoteSections = re.split(r'''(['"])''', stringToSearch)
    isStringIndexWithinQuote = isStringIndexWithinQuote = [False] * len(stringToSearch)
    if len(quoteSections) < 2:
        pass
    elif (len(quoteSections) - 1) % 2 != 0:
        raise Exception("Unexpected behavior of regex split. Please check your python version.")
    elif (len(quoteSections) - 1) % 4 != 0: #check re.split documentation to see how it works.
        raise Exception("Use of line breaks and nested quotes within quoted strings is not supported in Hybrid Fortran. \
Offending line: %s" %(stringToSearch))
    else:
        quoteSections.reverse()
        currSection = quoteSections.pop()
        index = len(currSection)
        if index > 0:
            isStringIndexWithinQuote[0:index] = [False] * len(currSection)
        while len(quoteSections) > 0:
            #opening quote part
            currSection = quoteSections.pop()
            sectionLength = len(currSection)
            prefIndex = 0
            if index > 0:
                prefIndex = index
            index = index + sectionLength
            if sectionLength != 1:
                raise Exception("Unexpected error: quote begin marker with strange number of characters")
            isStringIndexWithinQuote[prefIndex:index] = [True]

            #inbetween quotes part
            currSection = quoteSections.pop()
            sectionLength = len(currSection)
            prefIndex = index
            index = index + sectionLength
            isStringIndexWithinQuote[prefIndex:index] = [True] * sectionLength

            #closing quote part
            currSection = quoteSections.pop()
            sectionLength = len(currSection)
            prefIndex = index
            index = index + sectionLength
            if sectionLength != 1:
                raise Exception("Unexpected error: quote end marker with strange number of characters")
            isStringIndexWithinQuote[prefIndex:index] = [True]

            #next part that's not within quotes
            currSection = quoteSections.pop()
            sectionLength = len(currSection)
            prefIndex = index
            index = index + sectionLength
            isStringIndexWithinQuote[prefIndex:index] = [False] * sectionLength
        #sanity check
        if index != len(stringToSearch):
            raise Exception("Unexpected error: Index at the end of quotes search is %i. Expected: %i" %(index, len(stringToSearch)))
    return isStringIndexWithinQuote

def findRightMostOccurrenceNotInsideQuotes(stringToMatch, stringToSearch, rightStartAt=-1):
    indexesWithinQuotes = areIndexesWithinQuotes(stringToSearch)
    if rightStartAt > 0:
        nextRightStart = rightStartAt
    else:
        nextRightStart = len(stringToSearch)
    blankPos = -1
    while(True):
        blankPos = stringToSearch[:nextRightStart].rfind(stringToMatch)
        if blankPos <= 0 or not indexesWithinQuotes[blankPos]:
            break
        nextRightStart = blankPos
        blankPos = -1
    return blankPos

class BracketAnalyzer(object):
    currLevel = 0
    searchPattern = ""
    openingPattern = ""
    closingPattern = ""

    def __init__(self, openingChar="(", closingChar=")", pass_in_regex_pattern=False):
        self.currLevel = 0
        if pass_in_regex_pattern:
            self.searchPattern = re.compile(r"(.*?)(" + openingChar + r"|" + closingChar + r")(.*)", re.IGNORECASE)
            self.openingPattern = re.compile(openingChar, re.IGNORECASE)
            self.closingPattern = re.compile(closingChar, re.IGNORECASE)
        else:
            self.searchPattern = re.compile(r"(.*?)(" + re.escape(openingChar) + r"|" + re.escape(closingChar) + r")(.*)", re.IGNORECASE)
            self.openingPattern = re.compile(re.escape(openingChar), re.IGNORECASE)
            self.closingPattern = re.compile(re.escape(closingChar), re.IGNORECASE)

    @property
    def level(self):
        return self.currLevel

    def splitAfterClosingBrackets(self, string):
        work = string
        substring = ""
        match = self.searchPattern.match(work)
        if not match:
            return work, ""

        while match:
            if self.openingPattern.match(match.group(2)) != None:
                self.currLevel = self.currLevel + 1
            elif self.closingPattern.match(match.group(2)) != None:
                if self.currLevel == 0:
                    raise Exception("Closing bracket before opening one.")
                self.currLevel = self.currLevel - 1
            work = match.group(3)
            substring = substring + match.group(1) + match.group(2)
            if self.currLevel == 0:
                break
            match = self.searchPattern.match(work)
        if self.currLevel == 0:
            return substring, work
        else:
            return string, ""

    def currLevelAfterString(self, string):
        work = string
        match = self.searchPattern.match(work)
        while match:
            if self.openingPattern.match(match.group(2)) != None:
                self.currLevel += 1
            elif self.closingPattern.match(match.group(2)) != None:
                if self.currLevel == 0:
                    raise Exception("Closing bracket before opening one.")
                self.currLevel -= 1
            work = match.group(3)
            match = self.searchPattern.match(work)

        return self.currLevel
