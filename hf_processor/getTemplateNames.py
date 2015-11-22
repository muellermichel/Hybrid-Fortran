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

from xml.dom.minidom import Document
from DomHelper import parseString
from optparse import OptionParser
from GeneralHelper import openFile
import os
import sys
import traceback

##################### MAIN ##############################
#get all program arguments
parser = OptionParser()
parser.add_option("-c", "--callgraph", dest="callgraph",
                  help="analyzed callgraph XML file to read", metavar="XML")
parser.add_option("-d", "--debug", action="store_true", dest="debug",
                  help="show debug print in standard error output")
(options, args) = parser.parse_args()

if (not options.callgraph):
    sys.stderr.write("callgraph option is mandatory. Use '--help' for informations on how to use this module\n")
    sys.exit(1)

#read in callgraph xml
cgFile = openFile(str(options.callgraph),'rw')
data = cgFile.read()
cgDoc = parseString(data)
cgFile.close()

try:
  templateNames = [templateNode.getAttribute("name") for templateNode in cgDoc.getElementsByTagName("implementationTemplate")]
  print " ".join(templateNames)

except Exception, e:
  sys.stderr.write('Error when trying to extract template names: %s\n%s\n' \
    %(str(e), traceback.format_exc()))
  sys.exit(1)
