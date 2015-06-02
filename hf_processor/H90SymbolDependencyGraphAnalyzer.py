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
#  Procedure        H90CallGraphParser.py                              #
#  Comment          Generates a Fortran callgraph in xml format        #
#                   including parallel region annotations.             #
#                   For parsing it uses a combination of               #
#                   a finite state machine and regex (per line)        #
#  Date             2012/07/27                                         #
#  Author           Michel Müller (AOKI Laboratory)                    #
#**********************************************************************#


from xml.dom.minidom import Document
from DomHelper import addCallers, addCallees, createOrGetFirstNodeWithName

class SymbolDependencyAnalyzer:
    doc = None
    symbols = None
    callGraphEdgesByCallerName = None
    routinesByName = None
    callsByCallerName = None
    callsByCalleeName = None

    def __init__(self, doc):
        def getCallIndexByAttribute(attributeName):
            index = {}
            for call in doc.getElementsByTagName("call"):
                identifier = call.getAttribute(attributeName)
                callList = index.get(identifier, [])
                callList.append(call)
                index[identifier] = callList
            return index

        routinesByName = dict(
            (routine.getAttribute("name"), routine)
            for routine in doc.getElementsByTagName("routine")
        )
        callsByCalleeName = getCallIndexByAttribute("callee")
        callsByCallerName = getCallIndexByAttribute("caller")
        callGraphEdgesByCallerName = {}
        callGraphEdgesByCalleeName = {}

        def addCallees(routineName):
            for call in callsByCallerName.get(routineName, []):
                callerName = call.getAttribute("caller")
                if callerName != routineName:
                    raise Exception("unexpected error when constructing callgraph")
                calleeName = call.getAttribute("callee")
                edgeList = callGraphEdgesByCallerName.get(routineName, [])
                edgeList.append((callerName, calleeName))
                callGraphEdgesByCallerName[routineName] = edgeList
                edgeList = callGraphEdgesByCalleeName.get(calleeName, [])
                edgeList.append((callerName, calleeName))
                callGraphEdgesByCalleeName[calleeName] = edgeList
                addCallees(calleeName)

        callGraphRootRoutineNames = [
            routineName
            for routineName in routinesByName.keys()
            if len(callsByCalleeName.get(routineName, [])) == 0
        ]
        for routineName in callGraphRootRoutineNames:
            addCallees(routineName)

        #at this point we should have the complete callgraph
        self.callGraphEdgesByCallerName = callGraphEdgesByCallerName
        self.routinesByName = routinesByName
        self.callsByCalleeName = callsByCalleeName
        self.callsByCallerName = callsByCallerName
        self.doc = doc
        self.symbolsNode = createOrGetFirstNodeWithName('symbols', doc)

    def getAliasNamesByRoutineName(self, symbolName, routineName):
        aliasNamesByRoutineName = {routineName:symbolName}

        def addCallees(symbolName, routineName):
            for call in self.callsByCallerName.get(routineName, []):
                if call.getAttribute("caller") != routineName:
                    raise Exception(
                        "unexpected error when constructing callgraph for symbol aliases"
                    )
                argumentsInCall = [
                    argument.getAttribute("symbolName")
                    for argument in call.getElementsByTagName("argument")
                ]
                argIndex = -1
                try:
                    argIndex = argumentsInCall.index(symbolName)
                except Exception:
                    pass
                if argIndex < 0:
                    continue
                calleeName = call.getAttribute("callee")
                if calleeName in aliasNamesByRoutineName:
                    raise Exception(
                        "unsupported case: symbol %s is passed into routine %s in multiple paths in the callgraph" %(
                            symbolName, calleeName
                        )
                    )
                calleeRoutine = self.routinesByName[calleeName]
                argumentsInCallee = [
                    argument.getAttribute("symbolName")
                    for argument in calleeRoutine.getElementsByTagName("argument")
                ]
                if len(argumentsInCall) != len(argumentsInCallee):
                    raise Exception(
                        "call to %s in %s doesn't have the expected number of arguments" %(
                            calleeName,
                            routineName
                        )
                    )
                aliasName = argumentsInCallee[argIndex]
                aliasNamesByRoutineName[calleeName] = aliasName
                addCallees(aliasName, calleeName)

        def addCallers(symbolName, routineName):
            routine = self.routinesByName[routineName]
            argumentsInRoutine = [
                argument.getAttribute("symbolName")
                for argument in routine.getElementsByTagName("argument")
            ]
            argIndex = -1
            try:
                argIndex = argumentsInRoutine.index(symbolName)
            except Exception:
                pass
            if argIndex < 0:
                return
            calls = self.callsByCalleeName.get(routineName, [])
            if len(calls) > 1:
                raise Exception(
                    "unsupported case: symbol %s is passed into routine %s in multiple paths in the callgraph" %(
                        symbolName, routineName
                    )
                )
            if len(calls) == 0:
                return
            call = calls[0]
            if call.getAttribute("callee") != routineName:
                raise Exception(
                    "unexpected error when constructing callgraph for symbol aliases"
                )
            argumentsInCall = [
                argument.getAttribute("symbolName")
                for argument in call.getElementsByTagName("argument")
            ]
            callerName = call.getAttribute("caller")
            if len(argumentsInCall) != len(argumentsInRoutine):
                raise Exception(
                    "call to %s in %s doesn't have the expected number of arguments" %(
                        routineName,
                        callerName
                    )
                )
            aliasName = argumentsInCall[argIndex]
            aliasNamesByRoutineName[callerName] = aliasName
            addCallers(aliasName, calleeName)

        addCallees(symbolName, routineName)
        addCallers(symbolName, routineName)
        return aliasNamesByRoutineName

    def getSymbolDependencyGraph(self, symbolName, routineName):
        pass