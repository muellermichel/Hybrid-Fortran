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
#  Procedure        pretty.py                                          #
#  Date             2012/08/01                                         #
#  Author           Michel Müller (AOKI Laboratory)                    #
#**********************************************************************#

from xml.dom.minidom import Document
from DomHelper import parseString
from GeneralHelper import openFile, setupDeferredLogging
from optparse import OptionParser
import os
import sys
import logging

##################### MAIN ##############################
#get all program arguments
parser = OptionParser()
parser.add_option("-i", "--sourceXML", dest="source",
                  help="read callgraph from this XML file", metavar="XML")
(options, args) = parser.parse_args()

setupDeferredLogging('preprocessor.log', logging.DEBUG)

if (not options.source):
    logging.info("sourceXML option is mandatory. Use '--help' for informations on how to use this module")
    sys.exit(1)

#read in working xml
srcFile = openFile(str(options.source),'r')
data = srcFile.read()
srcFile.close()
doc = parseString(data)

sys.stdout.write(doc.toprettyxml())
