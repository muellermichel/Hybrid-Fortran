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

def containsKernels(routineNode):
	return False

def uniqueIdentifier(routineName, implementationName):
	return (routineName + "_hfauto_" + implementationName).strip()

class Routine(object):
	def __init__(self, name, routineNode, implementation):
		self.name = name
		self.implementation = implementation
		self._routineNode = routineNode
		self._specificationText = ""
		self._bodyText = ""
		self.sisterRoutine = None

	def __repr__(self):
		return self._text

	def nameInScope(self):
		if not self.sisterRoutine:
			return self.name
		return uniqueIdentifier(self.name, self.implementation.architecture[0])

	def loadHeaderLine(self, header):
		pass

	def loadSpecificationLine(self, specificationLine):
		pass

	def loadBodyLine(self, bodyLine):
		pass

	def synthesizedKernels(self):
		return self.implementation.synthesizedKernels()

	def implemented(self):
		return ""