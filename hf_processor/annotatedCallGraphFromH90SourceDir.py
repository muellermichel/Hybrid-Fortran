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
#  Procedure        annotatedCallGraphFromH90SourceDir.py              #
#  Comment          Generates a Fortran callgraph in xml format        #
#                   including parallel region annotations.             #
#  Date             2012/07/27                                         #
#  Author           Michel Müller (AOKI Laboratory)                    #
#**********************************************************************#


from xml.dom.minidom import Document
from optparse import OptionParser
from RecursiveDirEntries import dirEntries
from GeneralHelper import printProgressIndicator, progressIndicatorReset
from H90CallGraphParser import H90XMLCallGraphGenerator, H90XMLSymbolDeclarationExtractor
import os
import sys
import fileinput
import pdb

##################### MAIN ##############################
#get all program arguments
parser = OptionParser()
parser.add_option("-i", "--sourceDirectory", dest="sourceDir",
                  help="read files recursively from DIR", metavar="DIR")
parser.add_option("-d", "--debug", action="store_true", dest="debug",
                  help="show debug print in standard error output")
parser.add_option("-p", "--pretty", action="store_true", dest="pretty",
                  help="make xml output pretty")
(options, args) = parser.parse_args()

if (not options.sourceDir):
    sys.stderr.write("sourceDirectory option is mandatory. Use '--help' for informations on how to use this module\n")
    sys.exit(1)

#prepare xml output
doc = Document()
callGraphRoot = doc.createElement("callGraph")
doc.appendChild(callGraphRoot)

filesInDir = dirEntries(str(options.sourceDir), True, 'h90')

#first pass: loop through all h90 files (hybrid fortran 90) in the current directory
#   and build the basic callgraph based on subprocedures and calls. Also parse @-directives for annotations.
progressIndicatorReset(sys.stderr)
for fileNum, fileInDir in enumerate(filesInDir):
    parser = H90XMLCallGraphGenerator(doc)
    parser.debugPrint = options.debug
    parser.processFile(fileInDir)
    if options.debug:
        sys.stderr.write("Callgraph generated for " + fileInDir + "\n")
    else:
        printProgressIndicator(sys.stderr, fileInDir, fileNum + 1, len(filesInDir), "Callgraph parsing")

#second pass: loop again through all h90 files and parse the @domainDependant symbol declarations flags
#   -> update the callgraph document with this information.
#   note: We do this, since for simplicity reasons, the declaration parser relies on the symbol names that
#   have been declared in @domainDependant direcrives. Since these directives come *after* the declaration,
#   we need a second pass
progressIndicatorReset(sys.stderr)
for fileNum, fileInDir in enumerate(filesInDir):
    parser = H90XMLSymbolDeclarationExtractor(doc)
    parser.debugPrint = options.debug
    parser.processFile(fileInDir)
    if options.debug:
        sys.stderr.write("Symbol declarations extracted for " + fileInDir + "\n")
    else:
        printProgressIndicator(sys.stderr, fileInDir, fileNum + 1, len(filesInDir), "Symbol parsing")
progressIndicatorReset(sys.stderr)

if (options.pretty):
	sys.stdout.write(doc.toprettyxml())
else:
	sys.stdout.write(doc.toxml())

