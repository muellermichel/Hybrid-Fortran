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
#  Procedure        generateF90fromH90AndAnalyzedCallGraph.py          #
#  Comment          Takes one h90 file and the associated complete     #
#                   callgraph and produces a compilable F90 file       #
#  Date             2012/08/01                                         #
#  Author           Michel Müller (AOKI Laboratory)                    #
#**********************************************************************#


from xml.dom.minidom import Document, parseString
from optparse import OptionParser
from H90CallGraphParser import H90toF90Printer, H90XMLSymbolDeclarationExtractor
from GeneralHelper import openFile
import os
import sys
import traceback
import FortranImplementation

##################### MAIN ##############################
#get all program arguments
parser = OptionParser()
parser.add_option("-i", "--sourceH90File", dest="sourceFile",
                  help="H90 file to read", metavar="H90")
parser.add_option("-c", "--callgraph", dest="callgraph",
                  help="analyzed callgraph XML file to read", metavar="XML")
parser.add_option("-d", "--debug", action="store_true", dest="debug",
                  help="show debug print in standard error output")
parser.add_option("-m", "--implementation", dest="implementation",
                  help="specify a FortranImplementation classname as specified in FortranImplementation.py", metavar="IMP")
(options, args) = parser.parse_args()

if (not options.sourceFile):
    sys.stderr.write("sourceH90File option is mandatory. Use '--help' for informations on how to use this module\n")
    sys.exit(1)

if (not options.callgraph):
    sys.stderr.write("callgraph option is mandatory. Use '--help' for informations on how to use this module\n")
    sys.exit(1)

if (not options.implementation):
  sys.stderr.write("implementation option is mandatory. Use '--help' for informations on how to use this module\n")
  sys.exit(1)

#read in callgraph xml
cgFile = openFile(str(options.callgraph),'rw')
data = cgFile.read()
cgDoc = parseString(data)
cgFile.close()

# #first pass: gather symbol declaration informations and store them in the callgraph dom tree
# try:
#   symbolDeclExtractor = H90XMLSymbolDeclarationExtractor(cgDoc)
#   symbolDeclExtractor.debugPrint = options.debug
#   symbolDeclExtractor.processFile(options.sourceFile)
#   #note: currently we only write this out for debugging purposes - it is not being read again by the parser.
#   if options.debug:
#     debugFile = openFile(options.callgraph.split(".xml")[0] + "_debug.xml", "w")
#     debugFile.write(cgDoc.toprettyxml())
#     debugFile.close()

# except Exception, e:
#   sys.stderr.write('Error when extracting symbol declaration information from H90 file %s: %s\n%s\n' \
#     %(str(options.sourceFile), str(e), traceback.format_exc()))
#   sys.exit(1)

try:
  implementationAttr = getattr(FortranImplementation, options.implementation)
  implementation = implementationAttr()
  f90printer = H90toF90Printer(cgDoc, implementation)
  f90printer.debugPrint = options.debug
  f90printer.processFile(options.sourceFile)
except Exception, e:
  sys.stderr.write('Error when generating F90 from H90 file %s: %s\n%s\n' \
    %(str(options.sourceFile), str(e), traceback.format_exc()))
  sys.exit(1)
