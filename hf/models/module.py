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
		self.moduleNode = moduleNode
		self._hostDataRoutines = []
		self._acceleratorDataRoutines = []

	def loadRoutine(self, routine):
		self._hostDataRoutines.append(routine)

	def getHeader(self):
		return "module %s" %(self.name)

	def getSpecification(self):
		pass

	def getRoutines(self):
		return "\n".join([
			"%s\n%s" %(routine.getHostVersion(), routine.getAcceleratorVersion())
			for routine in self._routines
		])

	def getFooter(self):
		return "end module %s" %(self.name)