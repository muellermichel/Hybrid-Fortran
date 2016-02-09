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

	def __init__(self, name):
		if not type(name) in [str, unicode] or name.strip() == "":
			raise Exception("no valid name passed when trying to initialize routine")
		self.name = name

	def nameInScope(self):
		return self.name

class AnalyzableRoutine(Routine):

	def __init__(self, name, routineNode, implementation):
		super(AnalyzableRoutine, self).__init__(name)
		if not routineNode:
			raise Exception("no definition passed when trying to initialize routine '%s'" %(name))
		if not implementation:
			raise Exception("no implementation passed when trying to initialize routine '%s'" %(name))
		self.name = name
		self.implementation = implementation
		self.sisterRoutine = None
		self.node = routineNode
		self._headerText = ""
		self._specificationText = ""
		self._regions = []

	def nameInScope(self):
		if not self.sisterRoutine:
			return self.name
		return uniqueIdentifier(self.name, self.implementation.architecture[0])

	def loadHeaderLine(self, headerLine):
		self._headerText += headerLine

	def loadSpecificationLine(self, specificationLine):
		self._specificationText += specificationLine

	def loadRegion(self, region):
		self._regions.append(region)

	def implemented(self):
		return _headerText \
			+ _specificationText \
			+ "\n".join([
				region.implemented() for region in self._regions
			])