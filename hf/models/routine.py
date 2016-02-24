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

from models.region import Region

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
		self._footerText = ""
		self._regions = []
		self._currRegion = None

	def nameInScope(self):
		if not self.sisterRoutine:
			return self.name
		return uniqueIdentifier(self.name, self.implementation.architecture[0])

	def createRegion(self):
		self._currRegion = Region(self)
		self._regions.append(self._currRegion)
		return self._currRegion

	def loadLine(self, line):
		stripped = line.strip()
		if stripped == "":
			return
		if not self._currRegion:
			self._headerText += stripped + "\n"
			return
		self._footerText += stripped + "\n"

	def implemented(self):
		implementedRoutineElements = [self._headerText.strip() + "\n"] \
			+ [region.implemented() for region in self._regions] \
			+ [self._footerText.strip()]
		purgedRoutineElements = [
			(index, text) for index, text in enumerate(implementedRoutineElements)
			if text != ""
		]
		return "\n".join([
			text
			for (index, text) in purgedRoutineElements
		])