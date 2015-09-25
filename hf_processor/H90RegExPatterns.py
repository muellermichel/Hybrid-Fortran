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
#  Procedure        H90RegExPatterns.py                                #
#  Comment          Stores compiled regex patterns needed for          #
#                   Hybrid Fortran parsing                            #
#  Date             2013/01/29                                         #
#  Author           Michel Müller (RIKEN)                              #
#**********************************************************************#
import re

class H90RegExPatterns:
    dynamicPatternsByRegex = None
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
        self.dynamicPatternsByRegex = {}
        for patternName in self.staticRegexByPatternName:
            setattr(self, patternName, re.compile(self.staticRegexByPatternName[patternName], re.IGNORECASE))

    def get(self, regex):
        pattern = self.dynamicPatternsByRegex.get(regex)
        if pattern == None:
            pattern = re.compile(regex, re.IGNORECASE)
            self.dynamicPatternsByRegex[regex] = pattern
        return pattern
