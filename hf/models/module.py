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

class Module(object):

	def __init__(self, name, moduleNode):
		self.name = name
		self.node = moduleNode
		self._lastRoutine = None
		self._routinesByNameAndImplementationClass = {}
		self._postTextByRoutine = {}
		self._headerText = ""
		self._footerText = ""
		self._undecidedText = ""
		self._firstRoutinesByName = {}

	def nameInScope(self):
		return self.name

	def loadLine(self, line):
		stripped = line.strip()
		if stripped == "":
			return
		if not self._lastRoutine:
			self._headerText += stripped + "\n"
			return
		self._undecidedText += stripped + "\n"

	def loadRoutine(self, routine):
		if self._undecidedText != "":
			self._postTextByRoutine[self._lastRoutine.name] = self._undecidedText
			self._undecidedText = ""
		self._lastRoutine = routine
		routinesByImplementationClass = self._routinesByNameAndImplementationClass.get(routine.name, {})
		if len(routinesByImplementationClass.keys()) == 0:
			self._firstRoutinesByName[routine.name] = routine
		else:
			routine.sisterRoutine = self._firstRoutinesByName[routine.name]
		routinesByImplementationClass[routine.implementation.__class__.__name__] = routine
		self._routinesByNameAndImplementationClass[routine.name] = routinesByImplementationClass
		self._postTextByRoutine[routine.name] = ""

	def implemented(self):
		self._footerText = self._undecidedText
		self._undecidedText = ""

		routines = []
		for routine in sum([
			v.values()
			for _, v in self._routinesByNameAndImplementationClass.iteritems()
		], []):
			routines += routine.implementation.splitIntoCompatibleRoutines(routine)

		implementedModuleElements = \
			[self._headerText.strip()] \
			+ [
				(
					routine.implemented() + "\n" + self._postTextByRoutine[routine.name].strip()
				).strip()
				for routine in routines
			] \
			+ [self._footerText.strip()]
		purgedModuleElements = [
			text + "\n" for text in implementedModuleElements
			if text != ""
		]
		return '\n'.join([text for text in purgedModuleElements])