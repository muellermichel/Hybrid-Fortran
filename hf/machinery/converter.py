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

import os, sys, re, traceback, logging
from models.symbol import *
from models.routine import Routine, AnalyzableRoutine
from models.module import Module
from models.region import RegionType
from tools.metadata import *
from tools.commons import UsageError, BracketAnalyzer
from tools.analysis import SymbolDependencyAnalyzer, getAnalysisForSymbol, getArguments
from machinery.parser import H90CallGraphAndSymbolDeclarationsParser, getSymbolsByName, currFile, currLineNo
from machinery.commons import FortranCodeSanitizer, getSymbolAccessStringAndReminder

def getModuleArraysForCallee(calleeName, symbolAnalysisByRoutineNameAndSymbolName, symbolsByModuleNameAndSymbolName):
    moduleSymbols = []
    analysisBySymbolName = symbolAnalysisByRoutineNameAndSymbolName.get(calleeName, {})
    for symbolCallAnalysis in analysisBySymbolName.values():
        for symbolAnalysis in symbolCallAnalysis:
            if not symbolAnalysis.isModuleSymbol:
                continue
            symbol = symbolsByModuleNameAndSymbolName.get(symbolAnalysis.sourceModule, {}).get(symbolAnalysis.name)
            if symbol == None:
                #this happens for scalars for example
                continue
            symbol.analysis = symbolAnalysis
            moduleSymbols.append(symbol)
    return moduleSymbols

def getSymbolsByModuleNameAndSymbolName(cgDoc, moduleNodesByName, symbolAnalysisByRoutineNameAndSymbolName={}):
    symbolsByModuleNameAndSymbolName = {}
    for moduleName in moduleNodesByName.keys():
        moduleNode = moduleNodesByName.get(moduleName)
        if not moduleNode:
            continue
        symbolsByModuleNameAndSymbolName[moduleName] = getSymbolsByName(
            cgDoc,
            moduleNode,
            isModuleSymbols=True,
            symbolAnalysisByRoutineNameAndSymbolName=symbolAnalysisByRoutineNameAndSymbolName
        )
        for symbolName in symbolsByModuleNameAndSymbolName[moduleName]:
            symbol = symbolsByModuleNameAndSymbolName[moduleName][symbolName]
            symbol.sourceModule = moduleName
    return symbolsByModuleNameAndSymbolName

def getSymbolsByRoutineNameAndSymbolName(cgDoc, routineNodesByProcName, parallelRegionTemplatesByProcName, symbolAnalysisByRoutineNameAndSymbolName={}):
    symbolsByRoutineNameAndSymbolName = {}
    for procName in routineNodesByProcName:
        routine = routineNodesByProcName[procName]
        procName = routine.getAttribute('name')
        symbolsByRoutineNameAndSymbolName[procName] = getSymbolsByName(
            cgDoc,
            routine,
            parallelRegionTemplatesByProcName.get(procName,[]),
            isModuleSymbols=False,
            symbolAnalysisByRoutineNameAndSymbolName=symbolAnalysisByRoutineNameAndSymbolName
        )
    return symbolsByRoutineNameAndSymbolName

class H90toF90Converter(H90CallGraphAndSymbolDeclarationsParser):
    currentLineNeedsPurge = False
    tab_insideSub = "\t\t"
    tab_outsideSub = "\t"

    def __init__(
        self,
        cgDoc,
        implementationsByTemplateName,
        outputStream=sys.stdout,
        moduleNodesByName=None,
        parallelRegionData=None,
        symbolAnalysisByRoutineNameAndSymbolName=None,
        symbolsByModuleNameAndSymbolName=None,
        symbolsByRoutineNameAndSymbolName=None
    ):
        super(H90toF90Converter, self).__init__(
            cgDoc,
            moduleNodesByName=moduleNodesByName,
            parallelRegionData=parallelRegionData,
            implementationsByTemplateName=implementationsByTemplateName
        )
        self.outputStream = outputStream
        self.currSubroutineImplementationNeedsToBeCommented = False
        self.symbolsPassedInCurrentCallByName = {}
        self.additionalParametersByKernelName = {}
        self.additionalWrapperImportsByKernelName = {}
        self.currParallelIterators = []
        self.currRoutine = None
        self.currRegion = None
        self.currParallelRegion = None
        self.currModule = None
        self.currCallee = None
        self.currAdditionalSubroutineParameters = []
        self.currAdditionalCompactedSubroutineParameters = []
        self.codeSanitizer = FortranCodeSanitizer()
        self.currParallelRegionRelationNode = None
        self.currParallelRegionTemplateNode = None
        self.prepareLineCalledForCurrentLine = False
        self.preparedBy = None
        try:
            if symbolAnalysisByRoutineNameAndSymbolName != None:
                self.symbolAnalysisByRoutineNameAndSymbolName = symbolAnalysisByRoutineNameAndSymbolName
            else:
                symbolAnalyzer = SymbolDependencyAnalyzer(self.cgDoc)
                self.symbolAnalysisByRoutineNameAndSymbolName = symbolAnalyzer.getSymbolAnalysisByRoutine()
            if symbolsByModuleNameAndSymbolName != None:
                self.symbolsByModuleNameAndSymbolName = symbolsByModuleNameAndSymbolName
            else:
                self.symbolsByModuleNameAndSymbolName = getSymbolsByModuleNameAndSymbolName(self.cgDoc, self.moduleNodesByName, self.symbolAnalysisByRoutineNameAndSymbolName)

            if symbolsByRoutineNameAndSymbolName != None:
                self.symbolsByRoutineNameAndSymbolName = symbolsByRoutineNameAndSymbolName
            else:
                self.symbolsByRoutineNameAndSymbolName = getSymbolsByRoutineNameAndSymbolName(
                    self.cgDoc,
                    self.routineNodesByProcName,
                    self.parallelRegionTemplatesByProcName,
                    self.symbolAnalysisByRoutineNameAndSymbolName
                )
        except UsageError as e:
            logging.error('Error: %s' %(str(e)), extra={"hfLineNo":currLineNo, "hfFile":currFile})
            sys.exit(1)
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            logging.critical('Error when initializing h90 conversion: %s' %(str(e)), extra={"hfLineNo":currLineNo, "hfFile":currFile})
            logging.info(traceback.format_exc())
            sys.exit(1)

    def switchToNewRegion(self, regionClassName="Region", oldRegion=None):
        self.currRegion = self.currRoutine.createRegion(regionClassName, oldRegion)

    def endRegion(self):
        self.currRegion = None

    def prepareActiveParallelRegion(self, implementationFunctionName):
        routineNode = self.routineNodesByProcName.get(self.currRoutine.name)
        if not routineNode:
            raise Exception("no definition found for routine '%s'", self.currRoutine.name)
        if routineNode.getAttribute('parallelRegionPosition') != 'within':
            return False
        templates = self.parallelRegionTemplatesByProcName.get(self.currRoutine.name)
        if not templates or len(templates) == 0:
            raise Exception("Unexpected: no parallel template definition found for routine '%s'" \
                %(self.currRoutine.name))
        if len(templates) > 1 and self.implementation.multipleParallelRegionsPerSubroutineAllowed != True:
            raise Exception("Unexpected: more than one parallel region templates found for subroutine '%s' containing a parallelRegion directive \
This is not allowed for implementations using %s.\
                " %(
                    self.currRoutine.name,
                    type(self.implementation).__name__
                )
            )
        if implementationFunctionName == "parallelRegionBegin":
            self.switchToNewRegion("ParallelRegion")
            self.currParallelRegion = self.currRegion
        implementationAttr = getattr(self, 'implementation')
        functionAttr = getattr(implementationAttr, implementationFunctionName)
        self.prepareLine(functionAttr(self.currParallelRegionTemplateNode, self.branchAnalyzer.level), self.tab_insideSub)
        if implementationFunctionName == "parallelRegionEnd":
            self.switchToNewRegion(oldRegion=self.currParallelRegion)
            self.currParallelRegion = None
        return True

    def filterOutSymbolsAlreadyAliveInCurrentScope(self, symbolList):
        return [
            symbol for symbol in symbolList
            if not symbol.analysis \
            or ( \
                symbol.name not in self.symbolsByRoutineNameAndSymbolName.get(self.currRoutine.name, {}) \
                and symbol.analysis.argumentIndexByRoutineName.get(self.currRoutine.name, -1) == -1 \
            )
        ]

    def processSymbolMatchAndGetAdjustedLine(self, line, symbolMatch, symbol, isPointerAssignment):
        argumentString = symbolMatch.group(3)
        symbolAccessString, remainder = getSymbolAccessStringAndReminder(
            symbol,
            self.currParallelIterators,
            self.currParallelRegionTemplateNode,
            argumentString,
            self.currCallee,
            isPointerAssignment,
            isInsideParallelRegion=self.state == "inside_parallelRegion" or (
                self.state == "inside_branch"
                and self.stateBeforeBranch == "inside_parallelRegion"
            )
        )
        pattern1 = r"(.*?(?:\W|^))" + re.escape(symbol.nameInScope()) + re.escape(argumentString) + r"\s*"
        currMatch = self.patterns.get(pattern1).match(line)
        if not currMatch:
            pattern2 = r"(.*?(?:\W|^))" + re.escape(symbol.name) + re.escape(argumentString) + r"\s*"
            currMatch = self.patterns.get(pattern2).match(line)
            if not currMatch:
                raise Exception(\
                    "Symbol %s is accessed in an unexpected way. Note: '_d' postfix is reserved for internal use. Cannot match one of the following patterns: \npattern1: '%s'\npattern2: '%s'" \
                    %(symbol.name, pattern1, pattern2))
        prefix = currMatch.group(1)
        return (prefix + symbolAccessString + remainder).rstrip() + "\n"

    def processSymbolsAndGetAdjustedLine(self, line, isInsideSubroutineCall):
        isPointerAssignment = self.patterns.pointerAssignmentPattern.match(line) != None
        symbolNames = self.currSymbolsByName.keys()
        adjustedLine = line
        for symbolName in symbolNames:
            symbol = self.currSymbolsByName[symbolName]
            symbolWasMatched = False
            lineSections = []
            work = adjustedLine
            nextMatch = symbol.namePattern.match(work)
            while nextMatch:
                symbolWasMatched = True
                prefix = nextMatch.group(1)
                lineSections.append(prefix)
                postfix = nextMatch.group(3)
                processed = self.processSymbolMatchAndGetAdjustedLine(work, nextMatch, symbol, isPointerAssignment)
                adjustedMatch = symbol.namePattern.match(processed)
                if not adjustedMatch:
                    raise Exception("Symbol %s can't be matched again after adjustment. Adjusted portion: %s" %(symbol.name, processed))
                lineSections.append(adjustedMatch.group(2))
                work = adjustedMatch.group(3)
                nextMatch = symbol.namePattern.match(work)
            #whatever is left now as "work" is the unmatched trailer of the line
            lineSections.append(work)
            adjustedLine = ""
            for section in lineSections:
                adjustedLine = adjustedLine + section
            if not isInsideSubroutineCall:
                continue
            if symbolWasMatched:
                self.symbolsPassedInCurrentCallByName[symbolName] = symbol
        return adjustedLine.rstrip() + "\n"

    def processModuleSymbolImportAndGetAdjustedLine(self, line, symbols):
        if len(symbols) == 0:
            return line
        return self.implementation.adjustImportForDevice(
            line,
            symbols,
            RegionType.MODULE_DECLARATION,
            parallelRegionPosition=None,
            parallelRegionTemplates=[],
        )

    def processCallMatch(self, subProcCallMatch):
        super(H90toF90Converter, self).processCallMatch(subProcCallMatch)
        calleeNode = self.routineNodesByProcName.get(self.currCalleeName)
        if calleeNode:
            self.currCallee = AnalyzableRoutine(
                self.currCalleeName,
                calleeNode,
                self.parallelRegionTemplatesByProcName.get(self.currCalleeName),
                self.implementationForTemplateName(calleeNode.getAttribute('implementationTemplate'))
            )
            self.currCallee.loadArguments(getArguments(calleeNode))
        else:
            self.currCallee = Routine(self.currCalleeName)
        self.currRegion.loadCallee(self.currCallee)
        remainingCall = None
        if isinstance(self.currCallee, AnalyzableRoutine):
            additionalImports, additionalDeclarations = self.additionalParametersByKernelName.get(self.currCalleeName, ([], []))
            toBeCompacted = []
            toBeCompacted, declarationPrefix, notToBeCompacted = self.listCompactedSymbolsAndDeclarationPrefixAndOtherSymbols(additionalImports + additionalDeclarations)
            compactedArrayList = []
            if len(toBeCompacted) > 0:
                compactedArrayName = "hfimp_%s" %(self.currCalleeName)
                compactedArray = FrameworkArray(compactedArrayName, declarationPrefix, domains=[("hfauto", str(len(toBeCompacted)))], isOnDevice=True)
                compactedArrayList = [compactedArray]
            self.currCallee.loadAdditionalArgumentSymbols(sorted(notToBeCompacted + compactedArrayList))
            remainingCall = self.processSymbolsAndGetAdjustedLine(
                subProcCallMatch.group(2),
                isInsideSubroutineCall=True
            )
        else:
            remainingCall = subProcCallMatch.group(2)
        self.currCallee.loadArguments(self.currArguments)
        self.currRegion.loadPassedInSymbolsByName(self.symbolsPassedInCurrentCallByName)
        self.prepareLine("", self.tab_insideSub)
        if self.state != "inside_subroutine_call" and not (self.state == "inside_branch" and self.stateBeforeBranch == "inside_subroutine_call"):
            self.symbolsPassedInCurrentCallByName = {}
            self.currCallee = None

    def processModuleDeclarationLineAndGetAdjustedLine(self, line):
        baseline = line
        if self.currentLineNeedsPurge:
            baseline = "" #$$$ this seems dangerous
        adjustedLine = self.processModuleSymbolImportAndGetAdjustedLine(baseline, self.importsOnCurrentLine)
        if len(self.symbolsOnCurrentLine) > 0:
            adjustedLine = self.implementation.adjustDeclarationForDevice(
                adjustedLine,
                self.symbolsOnCurrentLine,
                declarationRegionType,
                self.currRoutine.node.getAttribute('parallelRegionPosition') if self.currRoutine else "inside"
            )
        return adjustedLine

    def processTemplateMatch(self, templateMatch):
        super(H90toF90Converter, self).processTemplateMatch(templateMatch)
        self.prepareLine("","")

    def processTemplateEndMatch(self, templateEndMatch):
        super(H90toF90Converter, self).processTemplateEndMatch(templateEndMatch)
        self.prepareLine("","")

    def processBranchMatch(self, branchMatch):
        super(H90toF90Converter, self).processBranchMatch(branchMatch)
        self.prepareLine("","")
        self.currentLineNeedsPurge = True

    def processModuleBeginMatch(self, moduleBeginMatch):
        super(H90toF90Converter, self).processModuleBeginMatch(moduleBeginMatch)
        self.implementation.processModuleBegin(self.currModuleName)
        self.currModule = Module(
            self.currModuleName,
            self.moduleNodesByName[self.currModuleName]
        )
        self.prepareLine(moduleBeginMatch.group(0), self.tab_outsideSub)

    def processModuleEndMatch(self, moduleEndMatch):
        self.prepareLine(moduleEndMatch.group(0), self.tab_outsideSub)
        self.outputStream.write(self.codeSanitizer.sanitizeLines(self.currModule.implemented() + "\n\n"))
        self.currModule = None
        self.implementation.processModuleEnd()
        super(H90toF90Converter, self).processModuleEndMatch(moduleEndMatch)

    def processProcBeginMatch(self, subProcBeginMatch):
        super(H90toF90Converter, self).processProcBeginMatch(subProcBeginMatch)
        self.currRoutine = self.currModule.createRoutine(
            self.currSubprocName,
            self.routineNodesByProcName.get(self.currSubprocName),
            self.parallelRegionTemplatesByProcName.get(self.currSubprocName),
            self.implementation
        )
        self.currRoutine.loadSymbolsByName(self.currSymbolsByName)
        self.currRoutine.loadArguments(self.currArguments)

        symbolsByUniqueNameToBeUpdated = {}

        #build list of additional subroutine parameters
        #(parameters that the user didn't specify but that are necessary based on the features of the underlying technology
        # and the symbols declared by the user, such us temporary arrays and imported symbols)
        additionalImportsForOurSelves, additionalDeclarationsForOurselves, additionalDummiesForOurselves = self.implementation.getAdditionalKernelParameters(
            self.cgDoc,
            self.currArguments,
            self.currRoutine.node,
            self.currRoutine.node,
            self.currModule.node,
            self.parallelRegionTemplatesByProcName.get(self.currRoutine.name),
            self.moduleNodesByName,
            self.currSymbolsByName,
            self.symbolAnalysisByRoutineNameAndSymbolName
        )
        logging.debug("additional symbols for ourselves;\nimports: %s\ndeclarations: %s\ndummies: %s" %(
            additionalImportsForOurSelves,
            additionalDeclarationsForOurselves,
            additionalDummiesForOurselves
        ), extra={"hfLineNo":currLineNo, "hfFile":currFile})
        for symbol in additionalImportsForOurSelves + additionalDeclarationsForOurselves:
            symbol.isEmulatingSymbolThatWasActiveInCurrentScope = True
        for symbol in additionalImportsForOurSelves + additionalDeclarationsForOurselves + additionalDummiesForOurselves:
            symbolsByUniqueNameToBeUpdated[symbol.uniqueIdentifier] = symbol
        toBeCompacted, declarationPrefix, otherImports = self.listCompactedSymbolsAndDeclarationPrefixAndOtherSymbols(
            additionalImportsForOurSelves + additionalDeclarationsForOurselves
        )
        compactedArrayList = []
        if len(toBeCompacted) > 0:
            compactedArrayName = "hfimp_%s" %(self.currRoutine.name)
            compactedArray = FrameworkArray(compactedArrayName, declarationPrefix, domains=[("hfauto", str(len(toBeCompacted)))], isOnDevice=True)
            compactedArrayList = [compactedArray]
        self.currAdditionalSubroutineParameters = sorted(otherImports + compactedArrayList)
        self.currRoutine.loadAdditionalArgumentSymbols(self.currAdditionalSubroutineParameters + additionalDummiesForOurselves)
        self.currAdditionalCompactedSubroutineParameters = sorted(toBeCompacted)

        #analyse whether this routine is calling other routines that have a parallel region within
        #+ analyse the additional symbols that come up there
        if self.currRoutine.node.getAttribute("parallelRegionPosition") == "inside":
            callsLibraries = self.cgDoc.getElementsByTagName("calls")
            if not callsLibraries or len(callsLibraries) == 0:
                raise Exception("Caller library not found.")
            calls = callsLibraries[0].getElementsByTagName("call")
            for call in calls:
                if call.getAttribute("caller") != self.currRoutine.name:
                    continue
                calleeName = call.getAttribute("callee")
                callee = self.routineNodesByProcName.get(calleeName)
                if not callee:
                    continue
                implementation = self.implementationForTemplateName(callee.getAttribute('implementationTemplate'))
                additionalImportsForDeviceCompatibility, \
                additionalDeclarationsForDeviceCompatibility, \
                additionalDummies = implementation.getAdditionalKernelParameters(
                    self.cgDoc,
                    getArguments(call),
                    self.currRoutine.node,
                    callee,
                    self.moduleNodesByName[callee.getAttribute('module')],
                    self.parallelRegionTemplatesByProcName.get(calleeName),
                    self.moduleNodesByName,
                    self.currSymbolsByName,
                    self.symbolAnalysisByRoutineNameAndSymbolName
                )
                for symbol in additionalImportsForDeviceCompatibility + additionalDeclarationsForDeviceCompatibility + additionalDummies:
                    symbol.resetScope(self.currRoutine.name)
                    symbolsByUniqueNameToBeUpdated[symbol.uniqueIdentifier] = symbol
                if 'DEBUG_PRINT' in implementation.optionFlags:
                    tentativeAdditionalImports = getModuleArraysForCallee(
                        calleeName,
                        self.symbolAnalysisByRoutineNameAndSymbolName,
                        self.symbolsByModuleNameAndSymbolName
                    )
                    additionalImports = self.filterOutSymbolsAlreadyAliveInCurrentScope(tentativeAdditionalImports)
                    additionalImportsByName = {}
                    for symbol in additionalImports:
                        additionalImportsByName[symbol.name] = symbol
                        symbol.resetScope(self.currRoutine.name)
                    self.additionalWrapperImportsByKernelName[calleeName] = additionalImportsByName.values()
                self.additionalParametersByKernelName[calleeName] = (additionalImportsForDeviceCompatibility, additionalDeclarationsForDeviceCompatibility + additionalDummies)
                logging.debug("call to %s; additional imports for device compatibility:" %(calleeName), extra={"hfLineNo":currLineNo, "hfFile":currFile})
                logging.debug("\n".join([str(symbol) for symbol in additionalImportsForDeviceCompatibility]), extra={"hfLineNo":currLineNo, "hfFile":currFile})
                logging.debug("call to %s; additional declarations for device compatibility:" %(calleeName), extra={"hfLineNo":currLineNo, "hfFile":currFile})
                logging.debug("\n".join([str(symbol) for symbol in additionalDeclarationsForDeviceCompatibility]), extra={"hfLineNo":currLineNo, "hfFile":currFile})
                logging.debug("call to %s; additinal dummy parameters:" %(calleeName), extra={"hfLineNo":currLineNo, "hfFile":currFile})
                logging.debug("\n".join([str(symbol) for symbol in additionalDummies]), extra={"hfLineNo":currLineNo, "hfFile":currFile})

        for symbolName in self.currSymbolsByName:
            symbol = self.currSymbolsByName[symbolName]
            if not symbol.uniqueIdentifier in symbolsByUniqueNameToBeUpdated:
                symbolsByUniqueNameToBeUpdated[symbol.uniqueIdentifier] = symbol

        self.currRoutine.loadAdditionalImportSymbols(symbolsByUniqueNameToBeUpdated.values())
        additionalImportsByScopedName = dict(
            (symbol.nameInScope(), symbol)
            for symbol in self.filterOutSymbolsAlreadyAliveInCurrentScope(
                sum(
                    [self.additionalParametersByKernelName[kernelName][0] for kernelName in self.additionalParametersByKernelName.keys()],
                    []
                ) + sum(
                    [self.additionalWrapperImportsByKernelName[kernelName] for kernelName in self.additionalWrapperImportsByKernelName.keys()],
                    []
                )
            )
        )
        logging.debug(
            "curr Module: %s; additional imports: %s" %(
                self.currModuleName,
                ["%s: %s from %s" %(symbol.name, symbol.declarationType, symbol.sourceModule) for symbol in additionalImportsByScopedName.values()]
            ),
            extra={"hfLineNo":currLineNo, "hfFile":currFile}
        )
        self.currRoutine.loadAdditionalImportSymbols(additionalImportsByScopedName.values())
        self.prepareLine("", self.tab_insideSub)

    def processProcExitPoint(self, line, isSubroutineEnd):
        if isSubroutineEnd:
            self.prepareLine("", self.tab_outsideSub)
        else:
            self.switchToNewRegion("RoutineEarlyExitRegion")
            self.prepareLine(line, self.tab_insideSub)
            self.switchToNewRegion()

    def processProcEndMatch(self, subProcEndMatch):
        self.endRegion()
        self.processProcExitPoint(subProcEndMatch.group(0), isSubroutineEnd=True)
        self.additionalParametersByKernelName = {}
        self.additionalWrapperImportsByKernelName = {}
        self.currAdditionalSubroutineParameters = []
        self.currAdditionalCompactedSubroutineParameters = []
        self.currSubroutineImplementationNeedsToBeCommented = False
        self.currRoutine = None
        super(H90toF90Converter, self).processProcEndMatch(subProcEndMatch)

    def processParallelRegionMatch(self, parallelRegionMatch):
        super(H90toF90Converter, self).processParallelRegionMatch(parallelRegionMatch)
        logging.debug(
            "...parallel region starts on line %i with active symbols %s" %(self.lineNo, str(self.currSymbolsByName.values())),
            extra={"hfLineNo":currLineNo, "hfFile":currFile}
        )
        self.switchToNewRegion("ParallelRegion")
        self.currParallelRegion = self.currRegion
        self.currParallelRegion.loadActiveParallelRegionTemplate(self.currParallelRegionTemplateNode)
        self.prepareLine("", self.tab_insideSub)

    def processParallelRegionEndMatch(self, parallelRegionEndMatch):
        super(H90toF90Converter, self).processParallelRegionEndMatch(parallelRegionEndMatch)
        self.prepareLine("", self.tab_insideSub)
        self.switchToNewRegion(oldRegion=self.currParallelRegion)
        self.currParallelRegion = None
        self.currParallelIterators = []
        self.currParallelRegionTemplateNode = None
        self.currParallelRegionRelationNode = None

    def processDomainDependantMatch(self, domainDependantMatch):
        super(H90toF90Converter, self).processDomainDependantMatch(domainDependantMatch)
        self.prepareLine("", "")

    def processDomainDependantEndMatch(self, domainDependantEndMatch):
        super(H90toF90Converter, self).processDomainDependantEndMatch(domainDependantEndMatch)
        self.prepareLine("", "")

    def processContainsMatch(self, containsMatch):
        super(H90toF90Converter, self).processContainsMatch(containsMatch)
        self.prepareLine(containsMatch.group(0), self.tab_outsideSub)

    def processInterfaceMatch(self, interfaceMatch):
        super(H90toF90Converter, self).processInterfaceMatch(interfaceMatch)
        self.prepareLine(interfaceMatch.group(0), self.tab_outsideSub)

    def processInterfaceEndMatch(self, interfaceEndMatch):
        super(H90toF90Converter, self).processContainsMatch(interfaceEndMatch)
        self.prepareLine(interfaceEndMatch.group(0), self.tab_outsideSub)

    def processNoMatch(self, line):
        super(H90toF90Converter, self).processNoMatch(line)
        self.prepareLine(line, "")

    def listCompactedSymbolsAndDeclarationPrefixAndOtherSymbols(self, additionalImports):
        toBeCompacted = []
        otherImports = []
        declarationPrefix = None
        for symbol in additionalImports:
            declType = symbol.declarationType

            # compact the imports of real type. Background: Experience has shown that too many
            # symbols passed to kernels, such that parameter list > 256Byte, can cause strange behavior. (corruption
            # of parameter list leading to launch failures)
            #-> need to pack all real symbols into an array to make it work reliably for cases where many reals are imported
            # why not integers? because they can be used as array boundaries.
            # Note: currently only a single real type per subroutine is supported for compaction
            currentDeclarationPrefix = symbol.getSanitizedDeclarationPrefix()
            if declType in [DeclarationType.FOREIGN_MODULE_SCALAR, DeclarationType.LOCAL_MODULE_SCALAR] \
            and 'real' in symbol.declarationPrefix.lower() \
            and (declarationPrefix == None or currentDeclarationPrefix == declarationPrefix):
                declarationPrefix = currentDeclarationPrefix
                symbol.isCompacted = True
                toBeCompacted.append(symbol)
            else:
                otherImports.append(symbol)
        return toBeCompacted, declarationPrefix, otherImports

    def processInsideModuleState(self, line):
        super(H90toF90Converter, self).processInsideModuleState(line)
        if self.state not in ['inside_module', 'inside_branch'] \
        or (self.state == 'inside_branch' and self.stateBeforeBranch != 'inside_module'):
            return
        if not self.prepareLineCalledForCurrentLine:
            self.prepareLine(self.processModuleDeclarationLineAndGetAdjustedLine(line), self.tab_outsideSub)

    def processInsideDeclarationsState(self, line):
        '''process everything that happens per h90 declaration line'''

        def finalizeDeclarationContext():
            ourSymbolsToAdd = sorted(
                self.currAdditionalSubroutineParameters + self.currAdditionalCompactedSubroutineParameters
            )
            additionalImports = []
            if 'DEBUG_PRINT' in self.implementation.optionFlags:
                callsLibraries = self.cgDoc.getElementsByTagName("calls")
                calls = callsLibraries[0].getElementsByTagName("call")
                for call in calls:
                    if call.getAttribute("caller") != self.currRoutine.name:
                        continue
                    calleeName = call.getAttribute("callee")
                    additionalImports += self.additionalWrapperImportsByKernelName.get(calleeName, [])
            packedRealSymbolsByCalleeName = {}
            compactionDeclarationPrefixByCalleeName = {}
            for calleeName in self.additionalParametersByKernelName.keys():
                additionalImports, additionalDeclarations = self.additionalParametersByKernelName[calleeName]
                toBeCompacted, declarationPrefix, _ = self.listCompactedSymbolsAndDeclarationPrefixAndOtherSymbols(
                    additionalImports + additionalDeclarations
                )
                if len(toBeCompacted) > 0:
                    compactionDeclarationPrefixByCalleeName[calleeName] = declarationPrefix
                    packedRealSymbolsByCalleeName[calleeName] = toBeCompacted
            self.currRegion.loadAdditionalContext(
                self.additionalParametersByKernelName,
                packedRealSymbolsByCalleeName,
                ourSymbolsToAdd,
                compactionDeclarationPrefixByCalleeName,
                self.currAdditionalCompactedSubroutineParameters
            )
            self.switchToNewRegion()

        subProcCallMatch = self.patterns.subprocCallPattern.match(line)
        parallelRegionMatch = self.patterns.parallelRegionPattern.match(line)
        domainDependantMatch = self.patterns.domainDependantPattern.match(line)
        subProcEndMatch = self.patterns.subprocEndPattern.match(line)
        templateMatch = self.patterns.templatePattern.match(line)
        templateEndMatch = self.patterns.templateEndPattern.match(line)
        branchMatch = self.patterns.branchPattern.match(line)

        if branchMatch:
            finalizeDeclarationContext()
            self.processBranchMatch(branchMatch)
            return
        if subProcCallMatch:
            finalizeDeclarationContext()
            self.switchToNewRegion("CallRegion")
            self.processCallMatch(subProcCallMatch)
            if (self.state == "inside_branch" and self.stateBeforeBranch != 'inside_subroutine_call') or (self.state != "inside_branch" and self.state != 'inside_subroutine_call'):
                self.processCallPost()
            self.switchToNewRegion()
            return
        if subProcEndMatch:
            finalizeDeclarationContext()
            self.processProcEndMatch(subProcEndMatch)
            if self.state == "inside_branch":
                self.stateBeforeBranch = 'inside_module_body'
            else:
                self.state = 'inside_module_body'
            return
        if parallelRegionMatch:
            raise UsageError("parallel region without parallel dependants")
        if self.patterns.subprocBeginPattern.match(line):
            raise UsageError("subprocedure within subprocedure not allowed")
        if templateMatch:
            raise UsageError("template directives are only allowed outside of subroutines")
        if templateEndMatch:
            raise UsageError("template directives are only allowed outside of subroutines")

        if domainDependantMatch:
            if self.state == "inside_branch":
                self.stateBeforeBranch = 'inside_domainDependantRegion'
            else:
                self.state = 'inside_domainDependantRegion'
            finalizeDeclarationContext()
            self.processDomainDependantMatch(domainDependantMatch)
            return

        importMatch1 = self.patterns.selectiveImportPattern.match(line)
        importMatch2 = self.patterns.singleMappedImportPattern.match(line)
        importMatch3 = self.patterns.importAllPattern.match(line)
        declarationMatch = self.patterns.symbolDeclPattern.match(line)
        specificationStatementMatch = self.patterns.specificationStatementPattern.match(line)
        if not ( \
            line.strip() == "" \
            or importMatch1 \
            or importMatch2 \
            or importMatch3 \
            or declarationMatch \
            or specificationStatementMatch \
        ):
            if self.state == "inside_branch":
                self.stateBeforeBranch = "inside_subroutine_body"
            else:
                self.state = "inside_subroutine_body"
            finalizeDeclarationContext()
            self.processInsideSubroutineBodyState(line)
            return

        self.analyseSymbolInformationOnCurrentLine(line)
        #we are never calling super and every match that would have prepared a line, would already have been covered
        #with a return -> safe to call prepareLine here.
        self.prepareLine(line, self.tab_insideSub)

    def processInsideSubroutineBodyState(self, line):
        '''process everything that happens per h90 subroutine body line'''
        branchMatch = self.patterns.branchPattern.match(line)
        if branchMatch:
            self.processBranchMatch(branchMatch)
            return

        if self.patterns.branchEndPattern.match(line):
            self.prepareLine("","")
            return

        subProcCallMatch = self.patterns.subprocCallPattern.match(line)
        if subProcCallMatch:
            self.switchToNewRegion("CallRegion")
            self.processCallMatch(subProcCallMatch)
            if self.state != 'inside_subroutine_call' and not (self.state == "inside_branch" and self.stateBeforeBranch == "inside_subroutine_call"):
                self.processCallPost()
            self.switchToNewRegion()
            return

        subProcEndMatch = self.patterns.subprocEndPattern.match(line)
        if subProcEndMatch:
            self.processProcEndMatch(subProcEndMatch)
            if self.state == "inside_branch":
                self.stateBeforeBranch = "inside_module_body"
            else:
                self.state = 'inside_module_body'
            return

        if self.patterns.earlyReturnPattern.match(line):
            self.processProcExitPoint(line, isSubroutineEnd=False)
            return

        if self.currSubroutineImplementationNeedsToBeCommented:
            self.prepareLine("! " + line, "")
            return

        parallelRegionMatch = self.patterns.parallelRegionPattern.match(line)
        if (parallelRegionMatch) \
        and self.currRoutine.node.getAttribute('parallelRegionPosition') == "within":
            templateRelations = self.parallelRegionTemplateRelationsByProcName.get(self.currRoutine.name)
            if templateRelations == None or len(templateRelations) == 0:
                raise Exception("No parallel region template relation found for this region.")
            for templateRelation in templateRelations:
                startLine = templateRelation.getAttribute("startLine")
                if startLine in [None, '']:
                    continue
                startLineInt = 0
                try:
                    startLineInt = int(startLine)
                except ValueError:
                    raise Exception("Invalid startLine definition for parallel region template relation: %s\n. All active template relations: %s\nRoutine node: %s" %(
                        templateRelation.toxml(),
                        [templateRelation.toxml() for templateRelation in templateRelations],
                        self.currRoutine.node.toprettyxml()
                    ))
                if startLineInt == self.lineNo:
                    self.currParallelRegionRelationNode = templateRelation
                    break
            else:
                raise Exception("No parallel region template relation was matched for the current linenumber.")
            logging.debug(
                "parallel region detected on line %i with template relation %s" %(self.lineNo, self.currParallelRegionRelationNode.toxml()),
                extra={"hfLineNo":currLineNo, "hfFile":currFile}
            )
            templates = self.parallelRegionTemplatesByProcName.get(self.currRoutine.name)
            if templates == None or len(templates) == 0:
                raise Exception("No parallel region template found for this region.")
            activeTemplateID = self.currParallelRegionRelationNode.getAttribute("id")
            for template in templates:
                if template.getAttribute("id") == activeTemplateID:
                    self.currParallelRegionTemplateNode = template
                    break
            else:
                raise Exception("No parallel region template has matched the active template ID.")
            self.currParallelIterators = self.implementation.getIterators(self.currParallelRegionTemplateNode)
            if len(self.currParallelIterators) > 0:
                self.processParallelRegionMatch(parallelRegionMatch)
                if self.state == "inside_branch":
                    self.stateBeforeBranch = "inside_parallelRegion"
                else:
                    self.state = 'inside_parallelRegion'
            else:
                self.prepareLine("","")
            return
        elif parallelRegionMatch:
            #this parallel region does not apply to us
            self.prepareLine("","")
            return

        if (self.patterns.parallelRegionEndPattern.match(line)):
            #note: this may occur when a parallel region is discarded because it doesn't apply
            #-> state stays within body and the region end line will trap here
            self.prepareLine("","")
            return

        domainDependantMatch = self.patterns.domainDependantPattern.match(line)
        if (domainDependantMatch):
            if self.state == "inside_branch":
                self.stateBeforeBranch = "inside_domainDependantRegion"
            else:
                self.state = 'inside_domainDependantRegion'
            self.processDomainDependantMatch(domainDependantMatch)
            return

        if (self.patterns.subprocBeginPattern.match(line)):
            raise Exception("subprocedure within subprocedure not allowed")

        adjustedLine = self.processSymbolsAndGetAdjustedLine(line, False)
        self.prepareLine(adjustedLine, self.tab_insideSub)

    def processInsideParallelRegionState(self, line):
        branchMatch = self.patterns.branchPattern.match(line)
        if branchMatch:
            self.processBranchMatch(branchMatch)
            return

        subProcCallMatch = self.patterns.subprocCallPattern.match(line)
        if subProcCallMatch:
            if subProcCallMatch.group(1) not in self.routineNodesByProcName.keys():
                message = self.implementation.warningOnUnrecognizedSubroutineCallInParallelRegion(
                    self.currRoutine.name,
                    subProcCallMatch.group(1)
                )
                if message != "":
                    logging.warning(message, extra={"hfLineNo":currLineNo, "hfFile":currFile})
            self.switchToNewRegion("CallRegion")
            self.processCallMatch(subProcCallMatch)
            if self.state != 'inside_subroutine_call' and not (self.state == "inside_branch" and self.stateBeforeBranch == "inside_subroutine_call"):
                self.processCallPost()
            self.switchToNewRegion()
            return

        parallelRegionEndMatch = self.patterns.parallelRegionEndPattern.match(line)
        if (parallelRegionEndMatch):
            self.processParallelRegionEndMatch(parallelRegionEndMatch)
            self.state = "inside_subroutine_body"
            if self.state == "inside_branch":
                self.stateBeforeBranch = "inside_subroutine_body"
            else:
                self.state = 'inside_subroutine_body'
            return

        if (self.patterns.parallelRegionPattern.match(line)):
            raise Exception("parallelRegion within parallelRegion not allowed")
        if (self.patterns.subprocEndPattern.match(line)):
            raise Exception("subprocedure end before @end parallelRegion")
        if (self.patterns.subprocBeginPattern.match(line)):
            raise Exception("subprocedure within subprocedure not allowed")

        adjustedLine = ""
        whileLoopMatch = self.patterns.whileLoopPattern.match(line)
        loopMatch = self.patterns.loopPattern.match(line)
        if whileLoopMatch == None and loopMatch != None:
            adjustedLine += self.implementation.loopPreparation().strip() + '\n'
        adjustedLine += self.processSymbolsAndGetAdjustedLine(line, isInsideSubroutineCall=False)
        self.prepareLine(adjustedLine, self.tab_insideSub)

    def processInsideDomainDependantRegionState(self, line):
        super(H90toF90Converter, self).processInsideDomainDependantRegionState(line)
        if self.state == "inside_domainDependantRegion":
            self.prepareLine("", "")

    def processInsideModuleDomainDependantRegionState(self, line):
        super(H90toF90Converter, self).processInsideModuleDomainDependantRegionState(line)
        if self.state == "inside_moduleDomainDependantRegion":
            self.prepareLine("", "")

    def processInsideBranch(self, line):
        super(H90toF90Converter, self).processInsideBranch(line)
        if self.state != "inside_branch":
            self.prepareLine("", "")

    def processInsideIgnore(self, line):
        super(H90toF90Converter, self).processInsideIgnore(line)
        self.prepareLine("", "")

    def processLine(self, line):
        self.currentLineNeedsPurge = False
        self.prepareLineCalledForCurrentLine = False
        super(H90toF90Converter, self).processLine(line)
        if not self.prepareLineCalledForCurrentLine:
            raise Exception(
                "Line has never been prepared - there is an error in the transpiler logic. Please contact the Hybrid Fortran maintainers. Parser state: %s; before branch: %s" %(
                    self.state,
                    self.stateBeforeBranch
                )
            )

    def processFile(self, fileName):
        self.outputStream.write(self.implementation.filePreparation(fileName))
        super(H90toF90Converter, self).processFile(fileName)

    def putLine(self, line):
        if line == "":
            return
        if self.currRegion:
            self.currRegion.loadLine(line, self.symbolsOnCurrentLine + self.importsOnCurrentLine)
        elif self.currRoutine:
            self.currRoutine.loadLine(line, self.symbolsOnCurrentLine + self.importsOnCurrentLine)
        elif self.currModule:
            self.currModule.loadLine(line)
        else:
            self.outputStream.write(self.codeSanitizer.sanitizeLines(line))

    #TODO: remove tab argument everywhere
    def prepareLine(self, line, tab):
        if self.prepareLineCalledForCurrentLine:
            raise Exception(
                "Line has already been prepared by %s - there is an error in the transpiler logic. Please contact the Hybrid Fortran maintainers. Parser state: %s; before branch: %s" %(
                    self.preparedBy,
                    self.state,
                    self.stateBeforeBranch
                )
            )
        import inspect
        self.preparedBy = inspect.getouterframes(inspect.currentframe(), 2)[1][3]
        self.prepareLineCalledForCurrentLine = True
        self.putLine(line)

    #TODO: remove tab argument everywhere
    def prepareAdditionalLine(self, line, tab, isInsertedBeforeCurrentLine=False):
        if not isInsertedBeforeCurrentLine and not self.prepareLineCalledForCurrentLine:
            raise Exception(
                "Line has not yet been prepared - there is an error in the transpiler logic. Please contact the Hybrid Fortran maintainers. Parser state: %s; before branch: %s" %(
                    self.state,
                    self.stateBeforeBranch
                )
            )
        self.putLine(line)
