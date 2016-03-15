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
from tools.patterns import RegExPatterns
from machinery.parser import H90CallGraphAndSymbolDeclarationsParser, getSymbolsByName, currFile, currLineNo
from machinery.commons import FortranCodeSanitizer

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
    patterns = RegExPatterns.Instance()
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
    patterns = RegExPatterns.Instance()
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
        self.currRoutineIsCallingParallelRegion = False
        self.currSubroutineImplementationNeedsToBeCommented = False
        self.symbolsPassedInCurrentCallByName = {}
        self.additionalParametersByKernelName = {}
        self.additionalWrapperImportsByKernelName = {}
        self.currParallelIterators = []
        self.currRoutine = None
        self.currRegion = None
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

    def switchToNewRegion(self):
        self.currRegion = self.currRoutine.createRegion()

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
                " %(self.currRoutine.name, type(self.implementation).__name__)
            )

        implementationAttr = getattr(self, 'implementation')
        functionAttr = getattr(implementationAttr, implementationFunctionName)
        self.prepareLine(functionAttr(self.currParallelRegionTemplateNode, self.branchAnalyzer.level), self.tab_insideSub)
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

    def processSymbolMatchAndGetAdjustedLine(self, line, symbolMatch, symbol, isInsideSubroutineCall, isPointerAssignment):
        def getAccessorsAndRemainder(accessorString):
            symbol_access_match = self.patterns.symbolAccessPattern.match(accessorString)
            if not symbol_access_match:
                return [], accessorString
            currBracketAnalyzer = BracketAnalyzer()
            return currBracketAnalyzer.getListOfArgumentsInOpenedBracketsAndRemainder(symbol_access_match.group(1))

        implementation = self.implementation
        if isInsideSubroutineCall and isinstance(self.currCallee, AnalyzableRoutine):
            implementation = self.currCallee.implementation

        #match the symbol's postfix again in the current given line. (The prefix could have changed from the last match.)
        postfix = symbolMatch.group(3)
        postfixEscaped = re.escape(postfix)
        accessors = []
        if len(symbol.domains) > 0: #0 domains could be an external function - need to retain postfix
            accessors, postfix = getAccessorsAndRemainder(postfix)

        if not implementation.supportsArbitraryDataAccessesOutsideOfKernels \
        and symbol.domains \
        and len(symbol.domains) > 0 \
        and not isInsideSubroutineCall \
        and not isPointerAssignment \
        and not symbol.isModuleSymbol \
        and not symbol.isHostSymbol \
        and self.state != "inside_parallelRegion" \
        and not (self.state == "inside_branch" and self.stateBeforeBranch == "inside_parallelRegion") \
        and self.currRoutine.node.getAttribute("parallelRegionPosition") != "outside" \
        and len(accessors) != 0 \
        and ( \
            not implementation.supportsNativeMemsetsOutsideOfKernels \
            or any([accessor.strip() != ":" for accessor in accessors]) \
        ):
            logging.warning(
                "Dependant symbol %s accessed with accessor domains (%s) outside of a parallel region or subroutine call in subroutine %s(%s:%i)" %(
                    symbol.name,
                    str(accessors),
                    self.currRoutine.name,
                    self.fileName,
                    self.lineNo
                ),
                extra={"hfLineNo":currLineNo, "hfFile":currFile}
            )

        #$$$ why are we checking for a present statement?
        #$$$ this should be replaced with generic access pattern matching
        accessPatternChangeRequired = False
        presentPattern = r"(.*?present\s*\(\s*)" + re.escape(symbol.nameInScope()) + postfixEscaped + r"\s*"
        currMatch = self.patterns.get(presentPattern).match(line)
        if not currMatch:
            pattern1 = r"(.*?(?:\W|^))" + re.escape(symbol.nameInScope()) + postfixEscaped + r"\s*"
            currMatch = self.patterns.get(pattern1).match(line)
            accessPatternChangeRequired = len(symbol.domains) > 0 #0 domains could be an external function call which we cannot touch
            if not currMatch:
                pattern2 = r"(.*?(?:\W|^))" + re.escape(symbol.name) + postfixEscaped + r"\s*"
                currMatch = self.patterns.get(pattern2).match(line)
                if not currMatch:
                    raise Exception(\
                        "Symbol %s is accessed in an unexpected way. Note: '_d' postfix is reserved for internal use. Cannot match one of the following patterns: \npattern1: '%s'\npattern2: '%s'" \
                        %(symbol.name, pattern1, pattern2))
        prefix = currMatch.group(1)
        numOfIndependentDomains = 0
        if accessPatternChangeRequired:
            numOfIndependentDomains = len(symbol.domains) - symbol.numOfParallelDomains
            offsets = []
            if len(accessors) != numOfIndependentDomains and len(accessors) != len(symbol.domains) and len(accessors) != 0:
                raise UsageError("Unexpected array access for symbol %s (%s): Please use either %i (number of parallel independant dimensions) \
    or %i (dimensions of loaded domain for this array) or zero accessors. Symbol Domains: %s; Symbol Init Level: %i; Parallel Region Position: %s; Parallel Active: %s; Symbol template:\n%s\n" %(
                    symbol.name,
                    str(accessors),
                    numOfIndependentDomains,
                    len(symbol.domains),
                    str(symbol.domains),
                    symbol.initLevel,
                    str(symbol.parallelRegionPosition),
                    symbol.parallelActiveDims,
                    symbol.template.toxml()
                ))
            if len(accessors) == 0 and (isInsideSubroutineCall or isPointerAssignment):
                for i in range(numOfIndependentDomains):
                    offsets.append(":")
            else:
                offsets += accessors

            iterators = self.currParallelIterators
            if isInsideSubroutineCall:
                calleeNode = self.routineNodesByProcName.get(self.currCalleeName)
                if calleeNode and calleeNode.getAttribute("parallelRegionPosition") != "outside":
                    iterators = []
        symbol_access = None
        if isPointerAssignment \
        or not accessPatternChangeRequired \
        or ( \
            self.state != "inside_parallelRegion" \
            and not (self.state == "inside_branch" and self.stateBeforeBranch == "inside_parallelRegion") \
            and not isInsideSubroutineCall \
            and not isPointerAssignment \
            and not symbol.isModuleSymbol \
            and not symbol.isHostSymbol \
            and len(accessors) == 0 \
        ):
            symbol_access = symbol.nameInScope()
        else:
            symbol_access = symbol.accessRepresentation(
                iterators,
                offsets,
                self.currParallelRegionTemplateNode,
                inside_subroutine_call=isInsideSubroutineCall
            )
        logging.debug(
            "symbol %s on line %i rewritten to %s; change required: %s, accessors: %s, num of independent domains: %i" %(
                str(symbol),
                self.lineNo,
                symbol_access,
                accessPatternChangeRequired,
                str(accessors),
                numOfIndependentDomains
            ),
            extra={"hfLineNo":currLineNo, "hfFile":currFile}
        )
        return (prefix + symbol_access + postfix).rstrip() + "\n"

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
                processed = self.processSymbolMatchAndGetAdjustedLine(work, nextMatch, symbol, isInsideSubroutineCall, isPointerAssignment)
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

    def processSymbolImportAndGetAdjustedLine(self, line, symbols):
        if len(symbols) == 0:
            return line
        return self.implementation.adjustImportForDevice(
            line,
            symbols,
            RegionType.KERNEL_CALLER_DECLARATION if self.currRoutineIsCallingParallelRegion else RegionType.OTHER,
            self.currRoutine.node.getAttribute('parallelRegionPosition'),
            self.parallelRegionTemplatesByProcName.get(self.currRoutine.name)
        )

    def processCallPostAndGetAdjustedLine(self, line):
        allSymbolsPassedByName = self.symbolsPassedInCurrentCallByName.copy()
        additionalImportsAndDeclarations = self.additionalParametersByKernelName.get(self.currCalleeName, ([],[]))
        additionalModuleSymbols = self.additionalWrapperImportsByKernelName.get(self.currCalleeName, [])
        for symbol in additionalImportsAndDeclarations[1] + additionalModuleSymbols:
            allSymbolsPassedByName[symbol.name] = symbol
        implementation = self.implementation
        if isinstance(self.currCallee, AnalyzableRoutine):
            implementation = self.currCallee.implementation
        adjustedLine = line + "\n" + implementation.kernelCallPost(allSymbolsPassedByName, self.currCallee.node)
        return adjustedLine

    def processCallMatch(self, subProcCallMatch):
        super(H90toF90Converter, self).processCallMatch(subProcCallMatch)
        calleeNode = self.routineNodesByProcName.get(self.currCalleeName)
        if calleeNode:
            self.currCallee = AnalyzableRoutine(
                self.currCalleeName,
                calleeNode,
                self.implementationForTemplateName(calleeNode.getAttribute('implementationTemplate'))
            )
        else:
            self.currCallee = Routine(self.currCalleeName)
        self.switchToNewRegion()

        adjustedLine = "call " + self.currCalleeName

        parallelRegionPosition = None
        if isinstance(self.currCallee, AnalyzableRoutine):
            parallelRegionPosition = self.currCallee.node.getAttribute("parallelRegionPosition")
        logging.debug(
            "In subroutine %s: Processing subroutine call to %s, parallel region position: %s" %(self.currRoutine.name, self.currCalleeName, parallelRegionPosition),
            extra={"hfLineNo":currLineNo, "hfFile":currFile}
        )
        if isinstance(self.currCallee, AnalyzableRoutine) and parallelRegionPosition == "within":
            parallelRegionTemplates = self.parallelRegionTemplatesByProcName.get(self.currCalleeName)
            if parallelRegionTemplates == None or len(parallelRegionTemplates) == 0:
                raise Exception("No parallel region templates found for subroutine %s" %(self.currCalleeName))
            adjustedLine = self.currCallee.implementation.kernelCallPreparation(parallelRegionTemplates[0], calleeNode=self.currCallee.node)
            adjustedLine = adjustedLine + "call " + self.currCalleeName + " " + self.currCallee.implementation.kernelCallConfig()

        # if isinstance(self.currCallee, AnalyzableRoutine) \
        # and getRoutineNodeInitStage(self.currCallee.node) == RoutineNodeInitStage.DIRECTIVES_WITHOUT_PARALLELREGION_POSITION:
        #     #special case. see also processBeginMatch.
        #     self.prepareLine("! " + subProcCallMatch.group(0), "")
        #     self.symbolsPassedInCurrentCallByName = {}
        #     self.currCallee = None
        #     return

        arguments = subProcCallMatch.group(2)
        paramListMatch = self.patterns.subprocFirstLineParameterListPattern.match(arguments)
        if not paramListMatch and len(arguments.strip()) > 0:
            raise Exception("Subprocedure arguments without enclosing brackets. This is invalid in Hybrid Fortran")

        additionalImports, additionalDeclarations = self.additionalParametersByKernelName.get(self.currCalleeName, ([], []))
        toBeCompacted = []
        toBeCompacted, declarationPrefix, notToBeCompacted = self.listCompactedSymbolsAndDeclarationPrefixAndOtherSymbols(additionalImports + additionalDeclarations)
        compactedArrayList = []
        if len(toBeCompacted) > 0:
            compactedArrayName = "hfimp_%s" %(self.currCalleeName)
            compactedArray = FrameworkArray(compactedArrayName, declarationPrefix, domains=[("hfauto", str(len(toBeCompacted)))], isOnDevice=True)
            compactedArrayList = [compactedArray]
        additionalSymbols = sorted(notToBeCompacted + compactedArrayList)
        if len(additionalSymbols) > 0:
            adjustedLine += "( &\n"
        else:
            adjustedLine += "("
        bridgeStr1 = " & !additional parameter"
        bridgeStr2 = "inserted by framework\n" + self.tab_insideSub + "& "
        remainingCall = self.processSymbolsAndGetAdjustedLine(
            paramListMatch.group(2),
            isInsideSubroutineCall=True
        ) if paramListMatch else ")\n"
        numOfProgrammerSpecifiedArguments = len(self.symbolsPassedInCurrentCallByName.keys())
        for symbolNum, symbol in enumerate(additionalSymbols):
            hostName = symbol.nameInScope()
            adjustedLine += hostName
            if symbolNum < len(additionalSymbols) - 1 or numOfProgrammerSpecifiedArguments > 0:
                adjustedLine += ", "
            if symbolNum < len(additionalSymbols) - 1 or paramListMatch:
                adjustedLine += "%s (type %i) %s" %(bridgeStr1, symbol.declarationType, bridgeStr2)
        adjustedLine += remainingCall

        callPreparationForSymbols = ""
        callPostForSymbols = ""
        if isinstance(self.currCallee, AnalyzableRoutine) \
        and self.state != "inside_subroutine_call" \
        and not (self.state == "inside_branch" and self.stateBeforeBranch == "inside_subroutine_call"):
            currSubprocNode = self.routineNodesByProcName.get(self.currRoutine.name)
            callPreparationForSymbols = ""
            callPostForSymbols = ""
            for symbol in self.symbolsPassedInCurrentCallByName.values():
                if symbol.isHostSymbol:
                    continue
                symbolsInCalleeByName = dict(
                    (symbol.name, symbol)
                    for symbol in self.symbolsByRoutineNameAndSymbolName.get(self.currCalleeName, {}).values()
                )
                symbolNameInCallee = None
                for symbolName in symbolsInCalleeByName:
                    analysis = getAnalysisForSymbol(self.symbolAnalysisByRoutineNameAndSymbolName, self.currCalleeName, symbolName)
                    if not analysis:
                        continue
                    if analysis.aliasNamesByRoutineName.get(self.currRoutine.name) == symbol.name:
                        symbolNameInCallee = symbolName
                        break
                if symbolNameInCallee == None:
                    continue #this symbol isn't passed in to the callee
                symbolInCallee = symbolsInCalleeByName.get(symbolNameInCallee)
                if symbolInCallee == None:
                    raise Exception("Symbol %s's data expected for callee %s, but could not be found" %(
                        symbolNameInCallee,
                        self.currCalleeName
                    ))
                callPreparationForSymbols += self.currCallee.implementation.callPreparationForPassedSymbol(
                    currSubprocNode,
                    symbolInCaller=symbol,
                    symbolInCallee=symbolInCallee
                )
                callPostForSymbols += self.currCallee.implementation.callPostForPassedSymbol(
                    currSubprocNode,
                    symbolInCaller=symbol,
                    symbolInCallee=symbolInCallee
                )
            adjustedLine = self.processCallPostAndGetAdjustedLine(adjustedLine)

        if self.state != "inside_subroutine_call" and not (self.state == "inside_branch" and self.stateBeforeBranch == "inside_subroutine_call"):
            self.symbolsPassedInCurrentCallByName = {}
            self.currCallee = None
            self.switchToNewRegion()

        self.prepareLine(callPreparationForSymbols + adjustedLine + callPostForSymbols, self.tab_insideSub)

    def processAdditionalSubroutineParametersAndGetAdjustedLine(self, line, additionalDummies):
        if len(self.currAdditionalSubroutineParameters + additionalDummies) == 0:
            return line

        paramListMatch = self.patterns.subprocFirstLineParameterListPattern.match(line)
        adjustedLine = None
        if paramListMatch:
            adjustedLine = paramListMatch.group(1) + " &\n" + self.tab_outsideSub + "& "
            paramListStr = paramListMatch.group(2).strip()
        else:
            adjustedLine = line + "( &\n" + self.tab_outsideSub + "& "
            paramListStr = ")"
        #adjusted line now contains only prefix, including the opening bracket
        symbolNum = 0
        for symbol in sorted(self.currAdditionalSubroutineParameters + additionalDummies):
            adjustedLine += symbol.nameInScope()
            if symbolNum != len(self.currAdditionalSubroutineParameters) - 1 or len(paramListStr) > 1:
                adjustedLine = adjustedLine + ","
            adjustedLine += " & !additional type %i symbol inserted by framework \n" %(symbol.declarationType) + self.tab_outsideSub + "& "
            symbolNum = symbolNum + 1
        return adjustedLine + paramListStr

    def processDeclarationLineAndGetAdjustedLine(self, line, declarationRegionType):
        baseline = line
        if self.currentLineNeedsPurge:
            baseline = ""
        adjustedLine = baseline

        for symbol in self.symbolsOnCurrentLine:
            match = symbol.getDeclarationMatch(str(adjustedLine))
            if not match:
                raise Exception("Symbol %s not found on a line where it has already been identified before. Current string to search: %s" \
                    %(symbol, adjustedLine))
            adjustedLine = symbol.getAdjustedDeclarationLine(match, declarationRegionType)

        if adjustedLine != baseline and declarationRegionType != RegionType.MODULE_DECLARATION:
            #$$$ this is scary. isn't there a better state test for this?
            adjustedLine = purgeDimensionAndGetAdjustedLine(adjustedLine, self.patterns).rstrip() + "\n"

        adjustedLine = self.processSymbolImportAndGetAdjustedLine(adjustedLine, self.importsOnCurrentLine)

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
            self.implementation
        )

        regionType = RegionType.KERNEL_CALLER_DECLARATION if self.currRoutineIsCallingParallelRegion else RegionType.OTHER
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
                if callee.getAttribute("parallelRegionPosition") != "within":
                    continue
                self.currRoutineIsCallingParallelRegion = True

        for symbolName in self.currSymbolsByName:
            symbol = self.currSymbolsByName[symbolName]
            if not symbol.uniqueIdentifier in symbolsByUniqueNameToBeUpdated:
                symbolsByUniqueNameToBeUpdated[symbol.uniqueIdentifier] = symbol
        for symbolName in symbolsByUniqueNameToBeUpdated:
            self.implementation.updateSymbolDeviceState(
                symbolsByUniqueNameToBeUpdated[symbolName],
                regionType,
                self.currRoutine.node.getAttribute("parallelRegionPosition")
            )
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
        allAdditionalImports = additionalImportsByScopedName.values()
        for symbol in allAdditionalImports:
            if symbol.declarationType not in [DeclarationType.FOREIGN_MODULE_SCALAR, DeclarationType.LOCAL_ARRAY, DeclarationType.MODULE_ARRAY]:
                continue
        adjustedLine = self.implementation.subroutinePrefix(self.currRoutine.node) \
            + " " \
            + self.processAdditionalSubroutineParametersAndGetAdjustedLine(
                subProcBeginMatch.group(0),
                additionalDummiesForOurselves
            ) \
            + '\n'
        if len(allAdditionalImports) > 0:
            adjustedLine += self.implementation.adjustImportForDevice(
                None,
                allAdditionalImports,
                RegionType.KERNEL_CALLER_DECLARATION if self.currRoutineIsCallingParallelRegion else RegionType.OTHER,
                self.currRoutine.node.getAttribute('parallelRegionPosition'),
                self.parallelRegionTemplatesByProcName.get(self.currRoutine.name)
            )
        self.prepareLine(adjustedLine + self.implementation.additionalIncludes(), self.tab_insideSub)

    def processProcExitPoint(self, line, is_subroutine_end):
        self.prepareLine(
            self.implementation.subroutineExitPoint(
                self.currSymbolsByName.values(), self.currRoutineIsCallingParallelRegion, is_subroutine_end
            ) + line,
            self.tab_outsideSub
        )

    def processProcEndMatch(self, subProcEndMatch):
        self.endRegion()
        self.processProcExitPoint(subProcEndMatch.group(0), is_subroutine_end=True)
        self.currRoutineIsCallingParallelRegion = False
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
        if self.prepareActiveParallelRegion('parallelRegionBegin'):
            self.switchToNewRegion()

    def processParallelRegionEndMatch(self, parallelRegionEndMatch):
        super(H90toF90Converter, self).processParallelRegionEndMatch(parallelRegionEndMatch)
        if self.prepareActiveParallelRegion('parallelRegionEnd'):
            self.switchToNewRegion()
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
            self.prepareLine(self.processDeclarationLineAndGetAdjustedLine(line, RegionType.MODULE_DECLARATION), self.tab_outsideSub)

    def processInsideDeclarationsState(self, line):
        def declarationEndStatements(declarationRegionType, isInsertedBeforeCurrentLine):
            additionalDeclarationsStr = ""

            #TODO $$$: most of the following code should probably be handled within implementation classes

            #########################################################################
            # gather additional symbols for ourselves                               #
            #########################################################################
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

            #########################################################################
            # gather symbols to be packed for called kernels                        #
            #########################################################################
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

            #########################################################################
            # mark in code, include additional symbols for kernel calls             #
            #########################################################################
            numberOfAdditionalDeclarations = ( \
                len(sum([self.additionalParametersByKernelName[kname][1] for kname in self.additionalParametersByKernelName], [])) \
                + len(ourSymbolsToAdd) \
                + len(packedRealSymbolsByCalleeName.keys()) \
            )
            if numberOfAdditionalDeclarations > 0:
                additionalDeclarationsStr = "\n" + self.tab_insideSub + \
                 "! ****** additional symbols inserted by framework to emulate device support of language features\n"

            defaultPurgeList = ['intent', 'public', 'parameter', 'allocatable']
            #########################################################################
            # create declaration lines for symbols for ourself                      #
            #########################################################################
            for symbol in ourSymbolsToAdd:
                purgeList = defaultPurgeList
                if not symbol.isCompacted:
                    purgeList=['public', 'parameter', 'allocatable']
                additionalDeclarationsStr += self.tab_insideSub + self.implementation.adjustDeclarationForDevice(
                    self.tab_insideSub +
                        symbol.getDeclarationLineForAutomaticSymbol(purgeList).strip(),
                    [symbol],
                    declarationRegionType,
                    self.currRoutine.node.getAttribute('parallelRegionPosition')
                ).rstrip() + " ! type %i symbol added for this subroutine\n" %(symbol.declarationType)
                logging.debug(
                    "...In subroutine %s: Symbol %s additionally declared" %(self.currRoutine.name, symbol),
                    extra={"hfLineNo":currLineNo, "hfFile":currFile}
                )

            #########################################################################
            # create declaration lines for called kernels                           #
            #########################################################################
            for calleeName in self.additionalParametersByKernelName:
                additionalImports, additionalDeclarations = self.additionalParametersByKernelName[calleeName]
                additionalImportSymbolsByName = {}
                for symbol in additionalImports:
                    additionalImportSymbolsByName[symbol.name] = symbol

                calleeNode = self.routineNodesByProcName.get(calleeName)
                implementation = self.implementationForTemplateName(calleeNode.getAttribute('implementationTemplate'))

                for symbol in self.filterOutSymbolsAlreadyAliveInCurrentScope(additionalDeclarations):
                    if symbol.declarationType not in [DeclarationType.LOCAL_ARRAY, DeclarationType.LOCAL_SCALAR]:
                        # only symbols that are local to the kernel actually need to be declared here.
                        # Everything else we should have in our own scope already, either through additional imports or
                        # through module association (we assume the kernel and its wrapper reside in the same module)
                        continue

                    logging.debug("...In subroutine %s: Symbol %s is required additionally for %s" %(self.currRoutine.name, symbol, calleeName))
                    #in case the array uses domain sizes in the declaration that are additional symbols themselves
                    #we need to fix them.
                    adjustedDomains = []
                    for (domName, domSize) in symbol.domains:
                        domSizeSymbol = additionalImportSymbolsByName.get(domSize)
                        if domSizeSymbol is None:
                            adjustedDomains.append((domName, domSize))
                            continue
                        adjustedDomains.append((domName, domSizeSymbol.nameInScope()))
                    symbol.domains = adjustedDomains
                    logging.debug("...In subroutine %s: Domains of Symbol %s adjusted to %s" %(self.currRoutine.name, symbol, adjustedDomains))

                    additionalDeclarationsStr += implementation.adjustDeclarationForDevice(
                        symbol.getDeclarationLineForAutomaticSymbol(defaultPurgeList).strip(),
                        [symbol],
                        declarationRegionType,
                        self.currRoutine.node.getAttribute('parallelRegionPosition')
                    ).rstrip() + " ! type %i symbol added for callee %s\n" %(symbol.declarationType, calleeName)
                #TODO: move this into implementation classes
                toBeCompacted = packedRealSymbolsByCalleeName.get(calleeName, [])
                if len(toBeCompacted) > 0:
                    #TODO: generalize for cases where we don't want this to be on the device (e.g. put this into Implementation class)
                    compactedArrayName = "hfimp_%s" %(calleeName)
                    compactedArray = FrameworkArray(
                        compactedArrayName,
                        compactionDeclarationPrefixByCalleeName[calleeName],
                        domains=[("hfauto", str(len(toBeCompacted)))],
                        isOnDevice=True
                    )
                    additionalDeclarationsStr += implementation.adjustDeclarationForDevice(
                        compactedArray.getDeclarationLineForAutomaticSymbol().strip(),
                        [compactedArray],
                        declarationRegionType,
                        self.currRoutine.node.getAttribute('parallelRegionPosition')
                    ).rstrip() + " ! compaction array added for callee %s\n" %(calleeName)
                    logging.debug(
                        "...In subroutine %s: Symbols %s packed into array %s" %(self.currRoutine.name, toBeCompacted, compactedArrayName),
                        extra={"hfLineNo":currLineNo, "hfFile":currFile}
                    )

            additionalDeclarationsStr += self.implementation.declarationEnd(
                self.currSymbolsByName.values() + additionalImports,
                self.currRoutineIsCallingParallelRegion,
                self.currRoutine.node,
                self.parallelRegionTemplatesByProcName.get(self.currRoutine.name)
            )

            #########################################################################
            # additional symbols for kernels to be packed                           #
            #########################################################################
            calleesWithPackedReals = packedRealSymbolsByCalleeName.keys()
            for calleeName in calleesWithPackedReals:
                for idx, symbol in enumerate(sorted(packedRealSymbolsByCalleeName[calleeName])):
                    #$$$ clean this up, the hf_imp prefix should be decided within the symbol class
                    additionalDeclarationsStr += "hfimp_%s(%i) = %s" %(calleeName, idx+1, symbol.nameInScope()) + \
                         " ! type %i symbol compaction for callee %s\n" %(symbol.declarationType, calleeName)

            #########################################################################
            # additional symbols for ourselves to be unpacked                       #
            #########################################################################
            #TODO: move this into implementation classes
            for idx, symbol in enumerate(self.currAdditionalCompactedSubroutineParameters):
                #$$$ clean this up, the hf_imp prefix should be decided within the symbol class
                additionalDeclarationsStr += "%s = hfimp_%s(%i)" %(symbol.nameInScope(), self.currRoutine.name, idx+1) + \
                         " ! additional type %i symbol compaction\n" %(symbol.declarationType)

            #########################################################################
            # mark the end of additional includes in code                           #
            #########################################################################
            if numberOfAdditionalDeclarations > 0:
                additionalDeclarationsStr += "! ****** end additional symbols\n\n"

            self.prepareAdditionalLine(additionalDeclarationsStr, self.tab_insideSub, isInsertedBeforeCurrentLine)
            self.switchToNewRegion()

        '''process everything that happens per h90 declaration line'''
        subProcCallMatch = self.patterns.subprocCallPattern.match(line)
        parallelRegionMatch = self.patterns.parallelRegionPattern.match(line)
        domainDependantMatch = self.patterns.domainDependantPattern.match(line)
        subProcEndMatch = self.patterns.subprocEndPattern.match(line)
        templateMatch = self.patterns.templatePattern.match(line)
        templateEndMatch = self.patterns.templateEndPattern.match(line)
        branchMatch = self.patterns.branchPattern.match(line)

        declarationRegionType = RegionType.OTHER
        if self.currRoutineIsCallingParallelRegion:
            declarationRegionType = RegionType.KERNEL_CALLER_DECLARATION

        if branchMatch:
            declarationEndStatements(declarationRegionType, isInsertedBeforeCurrentLine=True)
            self.processBranchMatch(branchMatch)
            return
        if subProcCallMatch:
            declarationEndStatements(declarationRegionType, isInsertedBeforeCurrentLine=True)
            self.processCallMatch(subProcCallMatch)
            if (self.state == "inside_branch" and self.stateBeforeBranch != 'inside_subroutine_call') or (self.state != "inside_branch" and self.state != 'inside_subroutine_call'):
                self.processCallPost()
            return
        if subProcEndMatch:
            declarationEndStatements(declarationRegionType, isInsertedBeforeCurrentLine=True)
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
            declarationEndStatements(declarationRegionType, isInsertedBeforeCurrentLine=True)
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
            declarationEndStatements(declarationRegionType, isInsertedBeforeCurrentLine=True)
            self.processInsideSubroutineBodyState(line)
            return

        self.analyseSymbolInformationOnCurrentLine(line)
        #we are never calling super and every match that would have prepared a line, would already have been covered
        #with a return -> safe to call prepareLine here.
        self.prepareLine(self.processDeclarationLineAndGetAdjustedLine(line, declarationRegionType), self.tab_insideSub)

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
            self.processCallMatch(subProcCallMatch)
            if self.state != 'inside_subroutine_call' and not (self.state == "inside_branch" and self.stateBeforeBranch == "inside_subroutine_call"):
                self.processCallPost()
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
            self.processProcExitPoint(line, is_subroutine_end=False)
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
            self.processCallMatch(subProcCallMatch)
            if self.state != 'inside_subroutine_call' and not (self.state == "inside_branch" and self.stateBeforeBranch == "inside_subroutine_call"):
                self.processCallPost()
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
        if self.currRegion:
            self.currRegion.loadLine(line)
        elif self.currRoutine:
            self.currRoutine.loadLine(line)
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
