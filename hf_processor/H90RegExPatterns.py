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
#  Procedure        H90RegExPatterns.py                                #
#  Comment          Stores compiled regex patterns needed for          #
#                   Hybrid Fortran parsing                            #
#  Date             2013/01/29                                         #
#  Author           Michel Müller (RIKEN)                              #
#**********************************************************************#
import re
from multiprocessing.connection import Client
from multiprocessing.connection import Listener
from GeneralHelper import Singleton

workAddress = ('localhost', 6000)
resultAddress = ('localhost', 6001)
resultConnection = None
resultListener = None
regexClient = None

class MatchEmulator(object):
    def __init__(self, text, matchgroups):
        self.groups = matchgroups
        self.text = text

    def group(self, groupNumber):
        if groupNumber == 0:
            return self.text
        if groupNumber > len(self.groups):
            raise Exception("cannot get group %i, pattern only seems to have %i groups" %(groupNumber, len(self.groups)))
        return self.groups[groupNumber - 1]

class PatternEmulator(object):
    def __init__(self, regex):
        global regexClient
        self.regex = regex
        regexClient.send({
            "regex": regex
        })
        self.__getResult()

    def __getResult(self):
        global resultConnection
        return resultConnection.recv()

    def match(self, text, flag=None):
        if flag not in [None, re.IGNORECASE]:
            raise Exception("flags other than IGNORECASE not implemented in match emulation")

        global regexClient
        regexClient.send({
            "regex": self.regex,
            "text": text
        })
        result = self.__getResult()
        matchgroups = result.get("matchgroups")
        if matchgroups == None:
            return None
        return MatchEmulator(text, matchgroups)

@Singleton
class H90RegExPatterns(object):
    staticRegexByPatternName = {
        'blankPattern': r'\s',
        'quotedStringPattern': r'''(["'])''',
        'subprocBeginPattern': r'\s*\w*\s*subroutine\s*(\w*).*',
        'subprocFirstLineParameterListPattern': r'(.*?\()(.*)',
        'subprocEndPattern': r'\s*end\s*subroutine.*',
        'subprocCallPattern': r'\s*call\s*(\w*)(.*)',
        'parallelRegionPattern': r'\s*@parallelRegion\s*{(.*)}.*',
        'domainDependantPattern': r'\s*@domainDependant\s*{(.*)}.*',
        'branchPattern': r'\s*@if\s*{(.*)}.*',
        'parallelRegionEndPattern': r'\s*@end\s*parallelRegion.*',
        'domainDependantEndPattern': r'\s*@end\s*domainDependant.*',
        'branchEndPattern': r'\s*@end\s*if.*',
        'intentPattern': r'.*?intent\s*\(\s*(in|out|inout)\s*\).*',
        'dimensionPattern': r'(.*?),?\s*dimension\s*\(\s*(.*?)\s*\)(.*)',
        'symbolDeclTestPattern': r'.*?::.*',
        'symbolDeclPattern': r"(\s*(?:double\s+precision|real|integer|character|logical)(?:.*?))\s*::(.*)",
        'pointerAssignmentPattern': r"^\s*\w+\s*\=\>\s*\w+.*",
        'whileLoopPattern': r"\s*do\s*while\W.*",
        'loopPattern': r"\s*do\W.*",
        'moduleBeginPattern': r'\s*module\s*(\w*).*',
        'moduleEndPattern': r'\s*end\s*module.*',
        'earlyReturnPattern': r'^\s*return(?:\s.*|$)',
        'templatePattern': r'\s*@scheme\s*{(.*)}.*',
        'templateEndPattern': r'\s*@end\s*scheme.*',
        'symbolAccessPattern': r'\s*\((.*)',
        'argumentPattern': r'\s*(?:subroutine|call)?\s*(?:\w*)\s*\((.*)'
    }

    def __init__(self):
        global resultConnection
        global resultListener
        global regexClient
        regexClient = Client(workAddress)
        resultListener = Listener(resultAddress)
        resultConnection = resultListener.accept()
        for patternName in self.staticRegexByPatternName:
            setattr(self, patternName, PatternEmulator(self.staticRegexByPatternName[patternName]))

    def __del__(self):
        global resultConnection
        global regexClient
        regexClient.close()
        resultConnection.close()

    def get(self, regex):
        return PatternEmulator(regex)
