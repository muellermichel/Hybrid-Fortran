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

from tools.commons import enum
from machinery.commons import ConversionOptions

RegionType = enum(
	"MODULE_DECLARATION",
	"KERNEL_CALLER_DECLARATION",
	"OTHER"
)

class Region(object):
	def __init__(self):
		self._text = ""

	def loadLine(self, line):
		stripped = line.strip()
		if stripped == "":
			return
		self._text += stripped + "\n"

	def implemented(self):
		stripped = self._text.strip()
		if stripped == "":
			return ""
		result = ""
		if ConversionOptions.Instance().debugPrint:
			result += "!<--- %s\n" %(type(self))
		result += stripped + "\n"
		if ConversionOptions.Instance().debugPrint:
			result += "!--->\n"
		return result

class ParallelRegion(Region):
	pass

class RoutineSpecificationRegion(Region):
	pass