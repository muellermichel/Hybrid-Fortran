#!/usr/bin/python
# -*- coding: UTF-8 -*-

# Copyright (C) 2013 Michel Müller, Rikagaku Kenkyuujo (RIKEN)

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
#  Procedure        FortranImplementation.py                           #
#  Comment          Put together valid fortran strings according to    #
#                   xml data and parser data                           #
#  Date             2012/08/02                                         #
#  Author           Michel Müller (AOKI Laboratory)                    #
#**********************************************************************#


from xml.dom.minidom import Document
from H90Symbol import Symbol, DeclarationType, splitDeclarationSettingsFromSymbols
from DomHelper import *
import os
import sys
import re
import pdb

def getErrorHandlingAfterKernelCall(calleeRoutineNode, errorVariable="cuerror", stopImmediately=True):
    name = calleeRoutineNode.getAttribute('name')
    if not name:
        raise Exception("Unexpected Error: routine node without name")
    stopLine = ""
    if stopImmediately:
        stopLine = "stop 1\n"
    return  "%s = cudaThreadSynchronize()\n" \
            "if(%s .NE. cudaSuccess) then\n"\
                "\twrite(0, *) 'CUDA error in kernel %s:', cudaGetErrorString(%s)\n%s" \
            "end if\n" %(errorVariable, errorVariable, name, errorVariable, stopLine)

#currently unused
def getTempDeallocationsAfterKernelCall(symbolsByName):
    result = ""
    symbolNames = symbolsByName.keys()
    for symbolName in symbolNames:
        symbol = symbolsByName[symbolName]
        if symbol.declarationType() != DeclarationType.LOCAL_ARRAY:
            continue
        result = result + "deallocate(%s)\n" %(symbol.automaticName())
    return result

def getDebugOffsetString(domainTuple):
    #$$$ change this - it must be consistant with storage_order.F90
    userdefinedDomNames = ["x", "y", "z", "nz", "i", "j", "vertical", "verticalPlus", "KMAX_CONST", "KMP1_CONST"]
    (dependantDomName, dependantDomSize) = domainTuple
    offset = ""
    if dependantDomName in userdefinedDomNames:
        offset = "DEBUG_OUT_%s" %(dependantDomName.strip())
    elif dependantDomSize in userdefinedDomNames:
        offset = "DEBUG_OUT_%s" %(dependantDomSize.strip())
    else:
        offset = "1"
    return offset

def getRuntimeDebugPrintStatements(symbolsByName, calleeRoutineNode):
    routineName = calleeRoutineNode.getAttribute('name')
    if not routineName:
        raise Exception("Callee routine name undefined.")
    result = "write(0,*) '*********** kernel %s finished *************** '\n" %(routineName)

    symbolNames = sorted(symbolsByName.keys())
    for symbolName in symbolNames:
        symbol = symbolsByName[symbolName]
        if not symbol.domains or len(symbol.domains) == 0:
            continue
        offsets = []
        for i in range(len(symbol.domains) - symbol.numOfParallelDomains):
            newDebugOffsetString = getDebugOffsetString(
                symbol.domains[i + symbol.numOfParallelDomains]
            )
            if newDebugOffsetString in offsets:
                #if there are multiple dimensions with the same sizes, use the second specified macro
                # - else we'd be always showing the diagonal of quadratic matrices.
                offsets.append(newDebugOffsetString + "_2")
            else:
                offsets.append(newDebugOffsetString)
        iterators = []
        for i in range(symbol.numOfParallelDomains):
            (dependantDomName, dependantDomSize) = symbol.domains[i]
            iterators.append(
                getDebugOffsetString(
                    symbol.domains[i]
                )
            )
        result = result + "cuTemp = %s\n" %(symbol.accessRepresentation(iterators, offsets))
        #michel 2013-4-18: the Fortran-style memcopy as used right now in the above line creates a runtime error immediately
        #                  if we'd like to catch such errors ourselves, we need to use the cuda API memcopy calls - however we
        #                  then also need information about the symbol size, which isn't available in the current implementation
        #                  (we currently look at the typedef just as a string).
        # result = result + "cuErrorMemcopy = cudaGetLastError()\n" \
        #     "if(cuErrorMemcopy .NE. cudaSuccess) then\n"\
        #         "\twrite(0, *) 'CUDA error when attempting to copy value from %s:', cudaGetErrorString(cuErrorMemcopy)\n" \
        #         "stop 1\n" \
        #     "end if\n" %(symbol.name)
        joinedDomains = iterators + offsets
        domainsStr = "(',"
        formStr = "'(A,"
        for i in range(len(joinedDomains)):
            if i != 0:
                domainsStr = domainsStr + ",', ',"
                formStr = formStr + ",A,"
            domainsStr = domainsStr + str(joinedDomains[i])
            formStr = formStr + "I3"
        formStr = formStr + ",A,E13.5)'"
        domainsStr = domainsStr + ",'):"
        result = result + "write(0,%s) '%s@%s', cuTemp\n" %(formStr, symbol.name, domainsStr)
    result = result + "write(0,*) '**********************************************'\n"
    result = result + "write(0,*) ''\n"
    return result

class FortranImplementation(object):

    def warningOnUnrecognizedSubroutineCallInParallelRegion(self, callerName, calleeName):
        return ""

    def kernelCallConfig(self):
        return ""

    def kernelCallPreparation(self, parallelRegionTemplate):
        return ""

    def kernelCallPost(self, symbolsByName, calleeRoutineNode):
        return ""

    def subroutinePrefix(self, routineNode):
        return ''

    def subroutineCallInvocationPrefix(self, subroutineName, parallelRegionTemplate):
        return 'call %s' %(subroutineName)

    def adjustImportForDevice(self, line, parallelRegionPosition):
        return line

    def adjustDeclarationForDevice(self, line, patterns, dependantSymbols, routineIsKernelCaller, parallelRegionPosition):
        return line

    def additionalIncludes(self):
        return ""

    def getAdditionalSubroutineSymbols(self, cgDoc, routineNode, parallelRegionTemplates):
        return [], []

    def extractListOfAdditionalSubroutineSymbols(self, routineNode, currSymbolsByName):
        return []

    def getIterators(self, parallelRegionTemplate):
        if not appliesTo(["CPU", ""], parallelRegionTemplate):
            return []

        (domainNames, domainSizes) = domainNamesAndSizesWithParallelRegionTemplate(parallelRegionTemplate)
        return domainNames

    def parallelRegionBegin(self, parallelRegionTemplate):
        (domainNames, domainSizes) = domainNamesAndSizesWithParallelRegionTemplate(parallelRegionTemplate)
        regionStr = ''
        for pos in range(len(domainNames)-1,-1,-1): #use inverted order (optimization of accesses for fortran storage order)
            domainName = domainNames[pos]
            regionStr = regionStr + 'do %s=1,%s' %(domainName, domainSizes[pos])
            if pos != 0:
                regionStr = regionStr + '; '
            pos = pos + 1
        return regionStr

    def parallelRegionEnd(self, parallelRegionTemplate):
        (domainNames, domainSizes) = domainNamesAndSizesWithParallelRegionTemplate(parallelRegionTemplate)
        pos = 0
        regionStr = ''
        for domainName in domainNames:
            regionStr = regionStr + 'end do'
            if pos != len(domainNames) - 1:
                regionStr = regionStr + '; '
            pos = pos + 1
        return regionStr

    def declarationEnd(self, dependantSymbols, routineIsKernelCaller, currRoutineNode, currParallelRegionTemplate):
        result = ""
        if currRoutineNode.getAttribute('parallelRegionPosition') != 'within':
            return result
        if currParallelRegionTemplate and appliesTo(["CPU", ""], currParallelRegionTemplate):
            iterators = self.getIterators(currParallelRegionTemplate)
            result = "integer(4) :: "
            for i in range(len(iterators)):
                if i != 0:
                    result = result + ", "
                result = result + iterators[i]
            result = result + "\n"

        return result

    def subroutineEnd(self, dependantSymbols, routineIsKernelCaller):
        return ''

class OpenMPFortranImplementation(FortranImplementation):
    def parallelRegionBegin(self, parallelRegionTemplate):
        openMPLines = "!$OMP PARALLEL\n!$OMP DO SCHEDULE(RUNTIME)\n"
        return openMPLines + FortranImplementation.parallelRegionBegin(self, parallelRegionTemplate)

    def parallelRegionEnd(self, parallelRegionTemplate):
        openMPLines = "\n!$OMP END DO\n!$OMP END PARALLEL\n"
        return FortranImplementation.parallelRegionEnd(self, parallelRegionTemplate) + openMPLines

    # def additionalIncludes(self):
    #     return "use omp_lib\n"

class CUDAFortranImplementation(FortranImplementation):

    def warningOnUnrecognizedSubroutineCallInParallelRegion(self, callerName, calleeName):
        return "WARNING: subroutine %s called inside %s's parallel region, but it is not defined in a h90 file.\n" \
                    %(calleeName, callerName)

    def kernelCallConfig(self):
        return "<<< cugrid, cublock >>>"

    def kernelCallPreparation(self, parallelRegionTemplate):
        if not appliesTo(["GPU"], parallelRegionTemplate):
            return ""

        blockSizePPNames = ["CUDA_BLOCKSIZE_X", "CUDA_BLOCKSIZE_Y", "CUDA_BLOCKSIZE_Z"]
        (domainNames, domainSizes) = domainNamesAndSizesWithParallelRegionTemplate(parallelRegionTemplate)
        if len(domainSizes) > 3 or len(domainSizes) < 1:
            raise Exception("Invalid number of parallel domains in parallel region definition.")

        gridStr = "cugrid = dim3("
        blockStr = "cublock = dim3("
        for i in range(3):
            if i != 0:
                gridStr = gridStr + ", "
                blockStr = blockStr + ", "
            if i < len(domainSizes):
                gridStr = gridStr + "ceiling(real(%s) / real(%s))" %(domainSizes[i], blockSizePPNames[i])
                blockStr = blockStr + "%s" %(blockSizePPNames[i])
            else:
                gridStr = gridStr + "1"
                blockStr = blockStr + "1"
        result = gridStr + ")\n" + blockStr + ")\n"
        return result

    def kernelCallPost(self, symbolsByName, calleeRoutineNode):
        if calleeRoutineNode.getAttribute('parallelRegionPosition') != 'within':
            return ""
        result = getErrorHandlingAfterKernelCall(calleeRoutineNode)
        #TODO: remove
        #result = result + getTempDeallocationsAfterKernelCall(symbolsByName)
        return result

    def getAdditionalSubroutineSymbols(self, cgDoc, routineNode, parallelRegionTemplates):
        additionalImports = []
        additionalDeclarations = []
        if not parallelRegionTemplates:
            return additionalImports, additionalDeclarations
        if not routineNode.getAttribute("parallelRegionPosition") == "within":
            return additionalImports, additionalDeclarations

        dependantTemplatesAndEntries = getDomainDependantTemplatesAndEntries(cgDoc, routineNode)
        for template, entry in dependantTemplatesAndEntries:
            dependantName = entry.firstChild.nodeValue
            symbol = Symbol(dependantName, template)
            symbol.loadDomainDependantEntryNodeAttributes(entry)

            #check for external module imports in kernel subroutines
            if symbol.sourceModule and symbol.sourceModule != "":
                symbol.loadRoutineNodeAttributes(routineNode, parallelRegionTemplates)
                symbol.isAutomatic = True
                additionalImports.append(symbol)
            #check for temporary arrays in kernel subroutines
            elif not symbol.intent or symbol.intent == "":
                symbol.loadRoutineNodeAttributes(routineNode, parallelRegionTemplates)
                if len(symbol.domains) == 0:
                    continue
                #at this point we know that the symbol is a temporary array -> need to store it
                symbol.isAutomatic = True
                additionalDeclarations.append(symbol)

        return sorted(additionalImports), sorted(additionalDeclarations)

    def extractListOfAdditionalSubroutineSymbols(self, routineNode, currSymbolsByName):
        result = []
        if not routineNode.getAttribute("parallelRegionPosition") == "within":
            return result

        for symbolName in currSymbolsByName.keys():
            symbol = currSymbolsByName[symbolName]
            if symbol.sourceModule and symbol.sourceModule != "":
                result.append(symbol)
            if (not symbol.intent or symbol.intent == "") and len(symbol.domains) > 0:
                #we got temporary arrays in a kernel -> need to be handled by framework
                result.append(symbol)
        return sorted(result)

    def getIterators(self, parallelRegionTemplate):
        if not appliesTo(["GPU"], parallelRegionTemplate):
            return []

        (domainNames, domainSizes) = domainNamesAndSizesWithParallelRegionTemplate(parallelRegionTemplate)
        return domainNames

    def subroutinePrefix(self, routineNode):
        parallelRegionPosition = routineNode.getAttribute("parallelRegionPosition")
        if not parallelRegionPosition or parallelRegionPosition == "" or parallelRegionPosition == "inside":
            return ""
        elif parallelRegionPosition == "within":
            return "attributes(global)"
        elif parallelRegionPosition == "outside":
            return "attributes(device)"
        else:
            raise Exception("invalid parallel region position defined for this routine: %s" %(parallelRegionPosition))

    def subroutineCallInvocationPrefix(self, subroutineName, parallelRegionTemplate):
        return 'call %s' %(subroutineName)

    def subroutineEnd(self, dependantSymbols, routineIsKernelCaller):
        deviceInitStatements = ""
        for symbol in dependantSymbols:
            if not symbol.isOnDevice:
                continue
            if symbol.isPresent:
                continue
            if (symbol.intent == "out" or symbol.intent == "inout") \
            and (routineIsKernelCaller or symbol.isToBeTransfered):
                symbol.isUsingDevicePostfix = False
                originalStr = symbol.selectAllRepresentation()
                symbol.isUsingDevicePostfix = True
                deviceStr = symbol.selectAllRepresentation()
                deviceInitStatements += originalStr + " = " + deviceStr + "\n"
        return deviceInitStatements

    def adjustImportForDevice(self, line, parallelRegionPosition):
        if parallelRegionPosition in ["within", "outside"]:
            return ""
        else:
            return line

    def adjustDeclarationForDevice(self, line, patterns, dependantSymbols, routineIsKernelCaller, parallelRegionPosition):
        if not dependantSymbols or len(dependantSymbols) == 0:
            raise Exception("no symbols to adjust")

        adjustedLine = line
        adjustedLine = adjustedLine.rstrip()

        declarationDirectivesWithoutIntent, declarationDirectives,  symbolDeclarationStr = splitDeclarationSettingsFromSymbols( \
            line, \
            dependantSymbols, \
            patterns, \
            withAndWithoutIntent=True \
        )

        #analyse state of symbols - already declared as on device or not?
        alreadyOnDevice = "undefined"
        for symbol in dependantSymbols:
            if not symbol.domains or len(symbol.domains) == 0:
                continue
            elif symbol.isPresent and alreadyOnDevice == "undefined":
                alreadyOnDevice = "yes"
            elif not symbol.isPresent and alreadyOnDevice == "undefined":
                alreadyOnDevice = "no"
            elif (symbol.isPresent and alreadyOnDevice == "no") or (not symbol.isPresent and alreadyOnDevice == "yes"):
                raise Exception("Declaration line contains a mix of device present, non-device-present arrays. \
Symbols vs present attributes:\n%s" %(str([(symbol.name, symbol.isPresent) for symbol in dependantSymbols])) \
                )
        copyHere = "undefined"
        for symbol in dependantSymbols:
            if not symbol.domains or len(symbol.domains) == 0:
                continue
            elif symbol.isToBeTransfered and copyHere == "undefined":
                copyHere = "yes"
            elif not symbol.isToBeTransfered and copyHere == "undefined":
                copyHere = "no"
            elif (symbol.isToBeTransfered and copyHere == "no") or (not symbol.isToBeTransfered and copyHere == "yes"):
                raise Exception("Declaration line contains a mix of transferHere / non transferHere arrays. \
Symbols vs transferHere attributes:\n%s" %(str([(symbol.name, symbol.transferHere) for symbol in dependantSymbols])) \
                )
        if copyHere == "yes" and alreadyOnDevice == "yes":
            raise Exception("Symbols with 'present' attribute cannot appear on the same specification line as symbols with 'transferHere' attribute.")


        #analyse the intent of the symbols. Since one line can only have one intent declaration, we can simply assume the intent of the
        #first symbol
        intent = dependantSymbols[0].intent
        #note: intent == None or "" -> is local array

        declarationType = dependantSymbols[0].declarationType()
        #packed symbols -> leave them alone
        if dependantSymbols[0].isCompacted:
            return adjustedLine + "\n"

        #module scalars in kernels
        if parallelRegionPosition == "within" \
        and (declarationType == DeclarationType.IMPORTED_SCALAR or declarationType == DeclarationType.MODULE_SCALAR):
            adjustedLine = declarationDirectives + " ,intent(in), value ::" + symbolDeclarationStr

        #local arrays in kernels
        elif parallelRegionPosition == "within" \
        and declarationType == DeclarationType.LOCAL_ARRAY:
            adjustedLine = declarationDirectives + ",intent(out), device ::" + symbolDeclarationStr

        #passed in scalars in kernels and inside kernels
        elif parallelRegionPosition in ["within", "outside"] \
        and len(dependantSymbols[0].domains) == 0 \
        and intent != "out":
            #handle scalars (passed by value)
            adjustedLine = declarationDirectives + " ,value ::" + symbolDeclarationStr
            for dependantSymbol in dependantSymbols:
                dependantSymbol.isOnDevice = True

        #arrays outside of kernels
        elif len(dependantSymbols[0].domains) > 0:
            if alreadyOnDevice == "yes" or not intent:
                # we don't need copies of the dependants on cpu
                adjustedLine = declarationDirectives + " ,device ::" + symbolDeclarationStr
                for dependantSymbol in dependantSymbols:
                    dependantSymbol.isOnDevice = True
            elif copyHere == "yes" or routineIsKernelCaller:
                for dependantSymbol in dependantSymbols:
                    dependantSymbol.isUsingDevicePostfix = True
                    dependantSymbol.isOnDevice = True
                    adjustedLine = str(adjustedLine) + "\n" + str(declarationDirectivesWithoutIntent) + " ,device :: " + str(dependantSymbol)

        return adjustedLine + "\n"

    def parallelRegionBegin(self, parallelRegionTemplate):
        (domainNames, domainSizes) = domainNamesAndSizesWithParallelRegionTemplate(parallelRegionTemplate)
        regionStr = ''
        if len(domainNames) > 3:
            raise Exception("Only up to 3 parallel dimensions supported. %i are specified: %s." %(len(domainNames), domainNames))

        cudaDims = ("x", "y", "z")

        for i in range(len(domainNames)):
            regionStr = regionStr + "%s = (blockidx%%%s - 1) * blockDim%%%s + threadidx%%%s\n" %(domainNames[i], cudaDims[i], cudaDims[i], cudaDims[i])

        regionStr = regionStr + "if ("
        for i in range(len(domainNames)):
            if i != 0:
                regionStr = regionStr + " .OR. "
            regionStr = regionStr + "%s > %s" %(domainNames[i], domainSizes[i])
        regionStr = regionStr + ") then\nreturn\nend if\n"

        return regionStr

    def parallelRegionEnd(self, parallelRegionTemplate):
        return ""

    def declarationEnd(self, dependantSymbols, routineIsKernelCaller, currRoutineNode, currParallelRegionTemplate):
        result = ""
        if currParallelRegionTemplate and appliesTo(["GPU"], currParallelRegionTemplate) and \
        currRoutineNode.getAttribute('parallelRegionPosition') == 'within':
            iterators = self.getIterators(currParallelRegionTemplate)
            result = "integer(4) :: "
            for i in range(len(iterators)):
                if i != 0:
                    result = result + ", "
                result = result + iterators[i]
            result = result + "\n"

        if routineIsKernelCaller:
            result = result + "type(dim3) :: cugrid, cublock\n"
            result = result + "integer(4) :: cuerror, cuErrorMemcopy\n"

        deviceInitStatements = ""
        for symbol in dependantSymbols:
            if not symbol.domains or len(symbol.domains) == 0:
                continue
            if not symbol.isOnDevice:
                continue
            if symbol.isPresent:
                continue
            if (symbol.intent == "in" or symbol.intent == "inout") \
            and (routineIsKernelCaller or symbol.isToBeTransfered):
                symbol.isUsingDevicePostfix = False
                originalStr = symbol.selectAllRepresentation()
                symbol.isUsingDevicePostfix = True
                deviceStr = symbol.selectAllRepresentation()
                deviceInitStatements += deviceStr + " = " + originalStr + "\n"
            elif routineIsKernelCaller or symbol.isToBeTransfered:
                deviceInitStatements += symbol.selectAllRepresentation() + " = 0\n"

        return result + deviceInitStatements

    def additionalIncludes(self):
        return "use cudafor\n"


class DebugCUDAFortranImplementation(CUDAFortranImplementation):

    def declarationEnd(self, dependantSymbols, routineIsKernelCaller, currRoutineNode, currParallelRegionTemplate):
        result = "real(8) :: cuTemp\n"
        result = result + CUDAFortranImplementation.declarationEnd(self, dependantSymbols, routineIsKernelCaller, \
            currRoutineNode, currParallelRegionTemplate)
        return result

    def kernelCallPost(self, symbolsByName, calleeRoutineNode):
        if calleeRoutineNode.getAttribute('parallelRegionPosition') != 'within':
            return ""
        result = getErrorHandlingAfterKernelCall(calleeRoutineNode, errorVariable="cuerror", stopImmediately=False)
        #michel 2013-4-18: Note, that doing the stop *after* trying to print the memory state hasn't really helped so far.
        #CUDA seems to fail any memcopy attempts after a kernel fails - maybe there is some method to clear the error state before
        #doing memcopy?
        result = result + getRuntimeDebugPrintStatements(symbolsByName, calleeRoutineNode)
        result = result + "if(cuerror .NE. cudaSuccess) then\n"\
                "\tstop 1\n" \
            "end if\n"
        #TODO: remove
        #result = result + getTempDeallocationsAfterKernelCall(symbolsByName)
        return result


class DebugEmulatedCUDAFortranImplementation(CUDAFortranImplementation):

    def __init__(self):
        self.currRoutineNode = None
        self.currDependantSymbols = None

    def declarationEnd(self, dependantSymbols, routineIsKernelCaller, currRoutineNode, currParallelRegionTemplate):
        self.currRoutineNode = currRoutineNode
        self.currDependantSymbols = dependantSymbols
        return CUDAFortranImplementation.declarationEnd(self, dependantSymbols, routineIsKernelCaller, currRoutineNode, currParallelRegionTemplate)

    def parallelRegionBegin(self, parallelRegionTemplate):
        result = CUDAFortranImplementation.parallelRegionBegin(self, parallelRegionTemplate)

        routineName = self.currRoutineNode.getAttribute('name')
        if not routineName:
            raise Exception("Routine name undefined.")

        iterators = self.getIterators(parallelRegionTemplate)
        if not iterators or len(iterators) == 0:
            raise Exception("No iterators in kernel.")

        conditional = "("
        for i in range(len(iterators)):
            if i != 0:
                conditional = conditional + " .AND. "
            conditional = conditional + "%s .EQ. %i" %(iterators[i], 1)
        conditional = conditional + ")"

        result = result + "if %s write(0,*) '*********** entering kernel %s finished *************** '\n" %(conditional, routineName)
        for symbol in self.currDependantSymbols:
            offsets = []
            for i in range(len(symbol.domains) - symbol.numOfParallelDomains):
                offsets.append("4")
            if symbol.intent == "in" or symbol.intent == "inout":
                result = result + "if %s then\n\twrite(0,*) '%s@1,1(,1)', %s\nend if\n" %(conditional, symbol.name, symbol.accessRepresentation(iterators, offsets))

        result = result + "if %s write(0,*) '**********************************************'\n" %(conditional)
        result = result + "if %s write(0,*) ''\n" %(conditional)

        return result


    def subroutineEnd(self, dependantSymbols, routineIsKernelCaller):
        self.currRoutineNode = None
        self.currDependantSymbols = None
        return CUDAFortranImplementation.subroutineEnd(self, dependantSymbols, routineIsKernelCaller)




