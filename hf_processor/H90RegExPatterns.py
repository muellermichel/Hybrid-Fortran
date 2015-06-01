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
    commentedPattern = None
    trailingCommentPattern = None
    subprocBeginPattern = None
    subprocParameterListPattern = None
    subprocEndPattern = None
    subprocCallPattern = None
    parallelRegionPattern = None
    domainDependantPattern = None
    parallelRegionEndPattern = None
    domainDependantEndPattern = None
    intentPattern = None
    dimensionPattern = None
    symbolDeclTestPattern = None
    symbolDeclPattern = None
    pointerAssignmentPattern = None
    loopPattern = None
    moduleBeginPattern = None
    moduleEndPattern = None
    earlyReturnPattern = None
    templatePattern = None
    templateEndPattern = None
    symbolAccessPattern = None
    argumentPattern = None

    def __init__(self):
        self.blankPattern = re.compile(r'\s', re.IGNORECASE)
        self.commentedPattern = re.compile(r'\s*\!', re.IGNORECASE)
        self.trailingCommentPattern = re.compile(r'(.*?)(\!.*)', re.IGNORECASE)
        self.quotedStringPattern = re.compile(r'''(["'])''', re.IGNORECASE)
        self.subprocBeginPattern = re.compile(r'\s*\w*\s*subroutine\s*(\w*).*', re.IGNORECASE)
        self.subprocFirstLineParameterListPattern = re.compile(r'(.*?\()(.*)')
        self.subprocEndPattern = re.compile(r'\s*end\s*subroutine.*', re.IGNORECASE)
        self.subprocCallPattern = re.compile(r'\s*call\s*(\w*)(.*)', re.IGNORECASE)
        self.parallelRegionPattern = re.compile(r'\s*@parallelRegion\s*{(.*)}.*', re.IGNORECASE)
        self.domainDependantPattern = re.compile(r'\s*@domainDependant\s*{(.*)}.*', re.IGNORECASE)
        self.branchPattern = re.compile(r'\s*@if\s*{(.*)}.*', re.IGNORECASE)
        self.parallelRegionEndPattern = re.compile(r'\s*@end\s*parallelRegion.*', re.IGNORECASE)
        self.domainDependantEndPattern = re.compile(r'\s*@end\s*domainDependant.*', re.IGNORECASE)
        self.branchEndPattern = re.compile(r'\s*@end\s*if.*', re.IGNORECASE)
        self.intentPattern = re.compile(r'.*?intent\s*\(\s*(in|out|inout)\s*\).*', re.IGNORECASE)
        self.dimensionPattern = re.compile(r'(.*?),?\s*dimension\s*\(\s*(.*?)\s*\)(.*)', re.IGNORECASE)
        self.symbolDeclTestPattern = re.compile(r'.*?::.*', re.IGNORECASE)
        self.symbolDeclPattern = re.compile(r"(\s*(?:double\s+precision|real|integer|character|logical)(?:.*?))\s*::(.*)", re.IGNORECASE)
        self.pointerAssignmentPattern = re.compile(r"^\s*\w+\s*\=\>\s*\w+.*", re.IGNORECASE)
        self.loopPattern = re.compile(r"\s*do\W.*", re.IGNORECASE)
        self.moduleBeginPattern = re.compile(r'\s*module\s*(\w*).*', re.IGNORECASE)
        self.moduleEndPattern = re.compile(r'\s*end\s*module.*', re.IGNORECASE)
        self.earlyReturnPattern = re.compile(r'^\s*return(?:\s.*|$)', re.IGNORECASE)
        self.templatePattern = re.compile(r'\s*@scheme\s*{(.*)}.*', re.IGNORECASE)
        self.templateEndPattern = re.compile(r'\s*@end\s*scheme.*', re.IGNORECASE)
        self.symbolAccessPattern = re.compile(r'\s*\((.*)', re.IGNORECASE)
        self.argumentPattern = re.compile(r'\s*,?\s*(\w+)\s*\(?(?:.*?)\)?\s*,?\s*(.*)', re.IGNORECASE)

