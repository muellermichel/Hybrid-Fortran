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

import re
import logging
from tools.commons import Singleton

@Singleton
class RegExPatterns:
    attributeRegex = r"\w*\s*(?:\(\s*[\w\,\s\:\+\-\*\/]*\s*(?:\(.*?\))?\s*\))?"
    dynamicPatternsByRegex = None
    staticRegexByPatternName = {
        'blankPattern': r'\s',
        'quotedStringPattern': r'''(["'])''',
        'subprocBeginPattern': r'\s*\w*\s*subroutine\s*(\w*).*',
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
        'multiSpecPattern': r'^(.*?)::(.*)',
        'doublePrecisionPattern': r'^\s*double\s+precision\s+(.*)',
        'standardTypePattern':r'^\s*((?:double\s+precision|real|integer|character|logical|complex)\s*(?:\(\s*[\w\,\s=*:]*\s*\))?).*',
        'dataStatementPattern': r'^\s*data\s+.*',
        'declarationKindPattern': r'(.*?)\s*kind\s*=\s*(\w*)\s*(.*)',
        'pointerAssignmentPattern': r"^\s*\w+\s*\=\>\s*\w+.*",
        'whileLoopPattern': r"^\s*do\s*while\W.*",
        'loopPattern': r"^\s*do\W.*",
        'interfacePattern': r"^\s*interface\s+.*",
        'interfaceEndPattern': r"^\s*end\s*interface\s+.*",
        'typePattern': r"^\s*type\s*\w+\s*$",
        'typeEndPattern': r"^\s*end\s*type\s*$",
        'moduleBeginPattern': r'^\s*module\s*(\w*).*',
        'moduleEndPattern': r'^\s*end\s*module.*',
        'earlyReturnPattern': r'^\s*return(?:\s.*|$)',
        'templatePattern': r'^\s*@scheme\s*{(.*)}.*',
        'templateEndPattern': r'^\s*@end\s*scheme.*',
        'symbolAccessPattern': r'\s*\((.*)',
        'symbolNamePattern': r'\s*(\w*)((?:\W.*)|\Z)',
        'argumentPattern': r'\s*(?:subroutine|call)?\s*(?:\w*)\s*\((.*)',
        'importPattern': r'^\s*use\s+(\w*)[,\s]*only\s*\:\s*([=>,\s\w]*)(?:\s.*|$)',
        'importAllPattern': r'^\s*use\s+(\w*)\s*$',
        'singleMappedImportPattern': r'\s*(\w*)\s*=>\s*(\w*)\s*',
        'callArgumentPattern': r'\s*(\w*)\s*(.*)',
        'containsPattern': r'\s*contains\s*',
        'routineNamePattern': r'(?:hfd_|hfk[0-9]*_)?(\w*)',
        'specificationStatementPattern': r"""
            ^\s*(
                procedure|external|intrinsic
                |public|private|allocatable|asynchronous
                |bind|data|dimension|intent|optional|parameter
                |pointer|protected|save
                |target|value|volatile|implicit
                |namelist|equivalence|common
            )\s+(.*)\s*$
        """
    }

    def __init__(self):
        self.dynamicPatternsByRegex = {}
        for patternName in self.staticRegexByPatternName:
            setattr(self, patternName, re.compile(self.staticRegexByPatternName[patternName], re.IGNORECASE | re.VERBOSE))

    def get(self, regex):
        pattern = self.dynamicPatternsByRegex.get(regex)
        if pattern == None:
            pattern = re.compile(regex, re.IGNORECASE | re.VERBOSE)
            self.dynamicPatternsByRegex[regex] = pattern
        return pattern
