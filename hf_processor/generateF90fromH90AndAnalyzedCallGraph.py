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
#  Procedure        generateF90fromH90AndAnalyzedCallGraph.py          #
#  Comment          Takes one h90 file and the associated complete     #
#                   callgraph and produces a compilable F90 file       #
#  Date             2012/08/01                                         #
#  Author           Michel Müller (AOKI Laboratory)                    #
#**********************************************************************#


from xml.dom.minidom import Document, parseString
from optparse import OptionParser
from H90CallGraphParser import H90toF90Printer
from GeneralHelper import openFile
import os
import sys
import json
import traceback
import FortranImplementation

def getDataFromFile(path):
	currFile = openFile(str(path),'rw')
	data = currFile.read()
	currFile.close()
	return data


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
									help="specify either a FortranImplementation classname or a JSON containing classnames by template name and a 'default' entry", metavar="IMP")
parser.add_option("--optionFlags", dest="optionFlags",
									help="can be used to switch on or off the following flags (comma separated): DO_NOT_TOUCH_GPU_CACHE_SETTINGS")
(options, args) = parser.parse_args()

optionFlags = [flag for flag in options.optionFlags.split(',') if flag not in ['', None]] if options.optionFlags != None else []
sys.stderr.write('Option Flags: %s\n' %(optionFlags))
if options.debug and 'DEBUG_PRINT' not in optionFlags:
	optionFlags.append('DEBUG_PRINT')

if (not options.sourceFile):
		sys.stderr.write("sourceH90File option is mandatory. Use '--help' for informations on how to use this module\n")
		sys.exit(1)

if (not options.callgraph):
		sys.stderr.write("callgraph option is mandatory. Use '--help' for informations on how to use this module\n")
		sys.exit(1)

if (not options.implementation):
	sys.stderr.write("implementation option is mandatory. Use '--help' for informations on how to use this module\n")
	sys.exit(1)

try:
	#read implementations
	implementationNamesByTemplateName=None
	try:
		implementationNamesByTemplateName=json.loads(getDataFromFile(options.implementation))
	except ValueError as e:
		sys.stderr.write('Error decoding implementation json (%s): %s\n' \
			%(str(options.implementation), str(e))
		)
		sys.exit(1)
	except Exception as e:
		sys.stderr.write('Could not interpret implementation parameter as json file to read. Trying to use it as an implementation name directly\n')
		implementationNamesByTemplateName={'default':options.implementation}
	if options.debug:
		sys.stderr.write('Initializing H90toF90Printer with the following implementations: %s\n' %(json.dumps(implementationNamesByTemplateName)))
	implementationsByTemplateName={
		templateName:getattr(FortranImplementation, implementationNamesByTemplateName[templateName])(optionFlags)
		for templateName in implementationNamesByTemplateName.keys()
	}
	#read in callgraph xml
	cgDoc = parseString(getDataFromFile(options.callgraph))
	f90printer = H90toF90Printer(cgDoc, implementationsByTemplateName)
	f90printer.debugPrint = options.debug
	f90printer.processFile(options.sourceFile)
except Exception as e:
	sys.stderr.write('Error when generating F90 from H90 file %s: %s\n%s\n' \
		%(str(options.sourceFile), str(e), traceback.format_exc())
	)
	sys.exit(1)
