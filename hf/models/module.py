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

from models.routine import AnalyzableRoutine

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

	@property
	def routines(self):
		return sum([
			v.values()
			for _, v in self._routinesByNameAndImplementationClass.iteritems()
		], [])

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

	def createRoutine(self, name, routineNode, parallelRegionTemplates, implementation):
		routine = AnalyzableRoutine(name, self, routineNode, parallelRegionTemplates, implementation)
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
		return routine

	def implemented(self):
		routines = []
		for routine in self.routines:
			routines += routine.implementation.generateRoutines(routine)

		for routine in routines:
			routine._checkParallelRegions()
			routine._updateSymbolReferences()
			routine._prepareAdditionalContext()
			routine._analyseSymbolUsage()

		self._footerText = self._undecidedText
		self._undecidedText = ""

		implementedModuleElements = \
			[self._headerText.strip()] \
			+ [
				(
					routine.implemented() + "\n" + self._postTextByRoutine.get(routine.name, "").strip()
				).strip()
				for routine in routines
			] \
			+ [self._footerText.strip()]
		purgedModuleElements = [
			text + "\n" for text in implementedModuleElements
			if text != ""
		]
		return '\n'.join([text for text in purgedModuleElements])

class ModuleStub(Module):
	def __init__(self, name):
		super(ModuleStub, self).__init__(name, None)