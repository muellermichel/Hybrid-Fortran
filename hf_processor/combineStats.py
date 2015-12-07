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

import pstats, sys
from optparse import OptionParser
from GeneralHelper import setupDeferredLogging
from RecursiveDirEntries import dirEntries
import logging

parser = OptionParser()
parser.add_option("-i", "--inputDirectory", dest="inputDir",
                  help="read files from DIR", metavar="DIR")
parser.add_option("-o", "--output", dest="output",
                  help="output combined statistics to FILENAME", metavar="FILENAME")
(options, args) = parser.parse_args()

setupDeferredLogging('preprocessor.log', logging.INFO)

if not options.inputDir or not options.output:
	logging.error("please see --help on how to use this program")
	sys.exit(1)

statFiles = dirEntries(str(options.inputDir), False, 'cprof')
logging.info("combining %s" %(str(statFiles)))
statistics = pstats.Stats(*statFiles)
statistics.dump_stats(options.output)