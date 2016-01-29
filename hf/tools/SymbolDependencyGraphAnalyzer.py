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

#**********************************************************************#
#  Procedure        CallGraphParser.py                              #
#  Comment          Generates a Fortran callgraph in xml format        #
#                   including parallel region annotations.             #
#                   For parsing it uses a combination of               #
#                   a finite state machine and regex (per line)        #
#  Date             2012/07/27                                         #
#  Author           Michel Müller (AOKI Laboratory)                    #
#**********************************************************************#


from xml.dom.minidom import Document
from tools.DomHelper import addCallers, addCallees, createOrGetFirstNodeWithName, getDomainDependantTemplatesAndEntries
from tools.GeneralHelper import enum, prettyprint, UsageError
import sys
import logging

SymbolType = enum(
    "UNDEFINED",
    "ARGUMENT_WITH_DOMAIN_DEPENDANT_SPEC",
    "ARGUMENT",
    "MODULE_DATA_USED_IN_CALLEE_GRAPH",
    "MODULE_DATA",
    "MODULE_DATA_USED_IN_CALLEE_GRAPH_WITH_DOMAIN_DEPENDANT_SPEC",
    "MODULE_DATA_WITH_DOMAIN_DEPENDANT_SPEC",
    "DOMAIN_DEPENDANT"
)

def emitSymbolAnalysisWarnings(analysisWarningsByCalleeName):
    for calleeName in analysisWarningsByCalleeName:
        warnings = analysisWarningsByCalleeName[calleeName]
        callers = [warning[0] for warning in warnings]
        logging.warning("Cannot fully analyze symbol dependencies since argument list from callers %s has different length (e.g. %i) than routine argument list (%i) for routine %s.\n" %(
                callers,
                warnings[0][1],
                warnings[0][2],
                calleeName
        ))

def getAnalysisForSymbol(symbolAnalysisByRoutineNameAndSymbolName, parentName, symbolName):
    symbolAnalysisPerCallee = symbolAnalysisByRoutineNameAndSymbolName.get(parentName, {}).get(symbolName, [])
    if len(symbolAnalysisPerCallee) > 0:
        return symbolAnalysisPerCallee[0]
    return None

def getArguments(parentNode):
    return [
        argument.getAttribute("symbolName")
        for argument in parentNode.getElementsByTagName("argument")
    ]

class SymbolAnalysis:
    def __init__(self):
        self.aliasNamesByRoutineName = {}
        self.argumentIndexByRoutineName = {}
        self.symbolType = SymbolType.UNDEFINED
        self.sourceModule = ""
        self.sourceSymbol = ""
        self.name = None

    def __unicode__(self):
        return unicode(self.__repr__())

    def __repr__(self):
        return "[Symbol %s (Type %s, from %s:%s); Aliases: %s; Argument Indices: %s]" %(
            self.name,
            str(self.symbolType),
            str(self.sourceModule),
            str(self.sourceSymbol),
            str(self.aliasNamesByRoutineName),
            str(self.argumentIndexByRoutineName)
        )

    @property
    def isModuleSymbol(self):
        return self.symbolType in [
            SymbolType.MODULE_DATA_USED_IN_CALLEE_GRAPH,
            SymbolType.MODULE_DATA,
            SymbolType.MODULE_DATA_USED_IN_CALLEE_GRAPH_WITH_DOMAIN_DEPENDANT_SPEC,
            SymbolType.MODULE_DATA_WITH_DOMAIN_DEPENDANT_SPEC
        ]

    @property
    def isDomainDependant(self):
        return self.symbolType in [
            SymbolType.ARGUMENT_WITH_DOMAIN_DEPENDANT_SPEC,
            SymbolType.MODULE_DATA_USED_IN_CALLEE_GRAPH_WITH_DOMAIN_DEPENDANT_SPEC,
            SymbolType.MODULE_DATA_WITH_DOMAIN_DEPENDANT_SPEC,
            SymbolType.DOMAIN_DEPENDANT
        ]


    def updateWith(self, analysis, routineName):
        for routineName in analysis.aliasNamesByRoutineName:
            self.aliasNamesByRoutineName[routineName] = analysis.aliasNamesByRoutineName[routineName]
        for routineName in analysis.argumentIndexByRoutineName:
            self.argumentIndexByRoutineName[routineName] = analysis.argumentIndexByRoutineName[routineName]
        if self.symbolType == SymbolType.UNDEFINED:
            self.symbolType = analysis.symbolType
        if (not type(self.sourceModule) in [unicode, str] or self.sourceModule == "") \
        and (type(analysis.sourceModule) in [unicode, str] and analysis.sourceModule != ""):
            logging.warning(
                "Symbol %s is imported from a module downstream in a callgraph (%s) while not being a module symbol earlier in the stream. SourceModule found where not expected.\n" %(
                    self.name,
                    routineName
                )
            )
        if (not type(self.sourceSymbol) in [unicode, str] or self.sourceSymbol == "") \
        and (type(analysis.sourceSymbol) in [unicode, str] and analysis.sourceSymbol != ""):
            logging.warning(
                "Symbol %s is imported from a module downstream in a callgraph (%s) while not being a module symbol earlier in the stream. SourceSymbol found where not expected.\n" %(
                    self.name,
                    routineName
                )
            )

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

        routinesByName = {}
        for routine in doc.getElementsByTagName("routine"):
            routineName = routine.getAttribute("name")
            if routineName in routinesByName:
                raise UsageError("Duplicate subroutines found with name %s. Subroutines need to have unique names in Hybrid Fortran." %(routineName))
            routinesByName[routineName] = routine

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

    def getSymbolAnalysisFor(self, routineName, symbolAnalysis=None, symbolAnalysisByNameAndSource=None, call=None, analysisWarningsByCalleeName=None):
        if symbolAnalysis == None:
            symbolAnalysis = {}
        if symbolAnalysisByNameAndSource == None:
            symbolAnalysisByNameAndSource = {}
        if analysisWarningsByCalleeName == None:
            analysisWarningsByCalleeName = {}
        routine = self.routinesByName.get(routineName)
        if not routine:
            return symbolAnalysis, symbolAnalysisByNameAndSource
        routineArguments = getArguments(routine)
        callArguments = []
        if call != None:
            callArguments = getArguments(call)
            if call.getAttribute("callee") != routineName:
                raise Exception("call passed to analysis of %s is not a call to this routine: %s" %(
                    routineName,
                    prettyprint(call)
                ))
            if len(callArguments) != len(routineArguments):
                warnings = analysisWarningsByCalleeName.get(routineName, [])
                warnings.append((call.getAttribute("caller"), len(callArguments), len(routineArguments), str(callArguments)))
                analysisWarningsByCalleeName[routineName] = warnings
                return symbolAnalysis, symbolAnalysisByNameAndSource

        templates_and_entries = getDomainDependantTemplatesAndEntries(
            self.doc,
            routine
        )
        analysisToAdd = {}

        #check arguments to this routine - temporarily remove previous calls from the analysis.
        #we don't want to overwrite information from previous calls.
        temporarilyStoredAnalysisByArgumentName = {}
        for argIndex, _ in enumerate(callArguments):
            localArgumentName = routineArguments[argIndex]
            if not (routineName, localArgumentName) in symbolAnalysis:
                continue
            temporarilyStoredAnalysisByArgumentName[localArgumentName] = symbolAnalysis[(routineName, localArgumentName)]
            del symbolAnalysis[(routineName, localArgumentName)]

        #Symbol Analysis based on local information
        for (_, entry) in templates_and_entries:
            analysis = SymbolAnalysis()
            analysis.name = entry.firstChild.nodeValue
            analysis.sourceModule = entry.getAttribute("sourceModule")
            analysis.sourceSymbol = entry.getAttribute("sourceSymbol")
            analysis.aliasNamesByRoutineName[routineName] = analysis.name
            argIndex = -1
            try:
                argIndex = routineArguments.index(analysis.name)
            except Exception:
                pass
            if argIndex > -1:
                analysis.argumentIndexByRoutineName[routineName] = argIndex
            if argIndex > -1:
                analysis.symbolType = SymbolType.ARGUMENT_WITH_DOMAIN_DEPENDANT_SPEC
            elif type(analysis.sourceModule) in [unicode, str] and analysis.sourceModule != "":
                analysis.symbolType = SymbolType.MODULE_DATA_WITH_DOMAIN_DEPENDANT_SPEC
            else:
                analysis.symbolType = SymbolType.DOMAIN_DEPENDANT
            analysisToAdd[analysis.name] = analysis
        for argIndex, argument in enumerate(routineArguments):
            if argument in analysisToAdd:
                continue
            analysis = SymbolAnalysis()
            analysis.name = argument
            analysis.symbolType = SymbolType.ARGUMENT
            analysis.argumentIndexByRoutineName[routineName] = argIndex
            analysisToAdd[analysis.name] = analysis

        #Symbol Analysis based on caller
        #--> check whether an analysis has already been done, update that and add it under the current routine, symbolName tuple
        for symbolName in analysisToAdd.keys():
            currentAnalysis = analysisToAdd[symbolName]
            argIndex = currentAnalysis.argumentIndexByRoutineName.get(routineName, -1)
            existingSymbolAnalysis = None
            if argIndex > -1 and len(callArguments) > 0:
                symbolNameInCaller = callArguments[argIndex]
                callerName = call.getAttribute("caller")
                existingSymbolAnalysis = symbolAnalysis.get((callerName, symbolNameInCaller), [])
            elif currentAnalysis.isModuleSymbol:
                existingModuleSymbolAnalysis = symbolAnalysisByNameAndSource.get((currentAnalysis.sourceSymbol, currentAnalysis.sourceModule))
                if existingModuleSymbolAnalysis:
                    existingSymbolAnalysis = [existingModuleSymbolAnalysis]
                else:
                    existingSymbolAnalysis = []
            else:
                existingSymbolAnalysis = [currentAnalysis]
            for analysis in existingSymbolAnalysis:
                if not isinstance(analysis, SymbolAnalysis):
                    raise Exception("unexpected error in routine %s: current analysis list contains unexpected objects: %s" %(
                        routineName,
                        str(existingSymbolAnalysis)
                    ))

            if len(existingSymbolAnalysis) == 0 or (len(existingSymbolAnalysis) == 1 and existingSymbolAnalysis[0] == currentAnalysis):
                symbolAnalysis[(routineName, symbolName)] = [currentAnalysis]
                if currentAnalysis.isModuleSymbol:
                    symbolAnalysisByNameAndSource[(currentAnalysis.sourceSymbol, currentAnalysis.sourceModule)] = currentAnalysis
            else:
                for analysis in existingSymbolAnalysis:
                    analysis.updateWith(currentAnalysis, routineName)
                symbolAnalysis[(routineName, symbolName)] = existingSymbolAnalysis

        #Analyse callgraph downstream from here
        for call in self.callsByCallerName.get(routineName, []):
            if call.getAttribute("caller") != routineName:
                raise Exception(
                    "unexpected error when constructing callgraph for symbol aliases"
                )
            calleeName = call.getAttribute("callee")
            try:
                symbolAnalysis, symbolAnalysisByNameAndSource = self.getSymbolAnalysisFor(
                    calleeName,
                    symbolAnalysis=symbolAnalysis,
                    symbolAnalysisByNameAndSource=symbolAnalysisByNameAndSource,
                    call=call,
                    analysisWarningsByCalleeName=analysisWarningsByCalleeName
                )
            except Exception as e:
                raise Exception(str(e) + " Caught in analysis for caller %s." %(routineName))
        for argumentName in temporarilyStoredAnalysisByArgumentName.keys():
            symbolAnalysis[(routineName, argumentName)] = temporarilyStoredAnalysisByArgumentName[argumentName] + symbolAnalysis.get((routineName, argumentName), [])

        return symbolAnalysis, symbolAnalysisByNameAndSource

    def getRootRoutines(self):
        return [
            routine
            for routine in self.routinesByName.values()
            if len(self.callsByCalleeName.get(routine.getAttribute("name"), [])) == 0
        ]

    def getSymbolAnalysis(self):
        symbolAnalysis = {}
        analysisWarningsByCalleeName = {}
        for routine in self.getRootRoutines():
            symbolAnalysis, _ = self.getSymbolAnalysisFor(
                routine.getAttribute("name"),
                symbolAnalysis=symbolAnalysis,
                analysisWarningsByCalleeName=analysisWarningsByCalleeName
            )
        emitSymbolAnalysisWarnings(analysisWarningsByCalleeName)
        return symbolAnalysis

    def getSymbolAnalysisForCallGraphStartingFrom(self, routineName):
        analysisWarningsByCalleeName = {}
        symbolAnalysis, _ = self.getSymbolAnalysisFor(
            routineName,
            symbolAnalysis=symbolAnalysis,
            analysisWarningsByCalleeName=analysisWarningsByCalleeName
        )
        emitSymbolAnalysisWarnings(analysisWarningsByCalleeName)
        return symbolAnalysis

    def getSymbolAnalysisByRoutine(self, startingFromRoutine=None):
        symbolAnalysis = None
        if startingFromRoutine:
            symbolAnalysis = self.getSymbolAnalysisForCallGraphStartingFrom(startingFromRoutine)
        else:
            symbolAnalysis = self.getSymbolAnalysis()
        symbolAnalysisByRoutine = {}
        for (routineName, symbolName) in symbolAnalysis.keys():
            if routineName in symbolAnalysisByRoutine:
                symbolAnalysisByRoutine[routineName][symbolName] = symbolAnalysis[(routineName, symbolName)]
            else:
                symbolAnalysisByRoutine[routineName] = {symbolName:symbolAnalysis[(routineName, symbolName)]}
        return symbolAnalysisByRoutine

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