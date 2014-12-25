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

def getVectorSizePPNames(parallelRegionTemplate):
    template = getTemplate(parallelRegionTemplate)
    template_prefix = ''
    if template != '':
        template_prefix = '_' + template
    return ["CUDA_BLOCKSIZE_X" + template_prefix, "CUDA_BLOCKSIZE_Y" + template_prefix, "CUDA_BLOCKSIZE_Z" + template_prefix]

def getIteratorDeclaration(currRoutineNode, currParallelRegionTemplates, architectures):
    result = ""
    iteratorsByName = {}
    if not currParallelRegionTemplates or not currRoutineNode.getAttribute('parallelRegionPosition') == 'within':
        return result
    for template in currParallelRegionTemplates:
        if not appliesTo(architectures, template):
            continue
        iteratorsByName.update(dict(
            (iterator, None)
            for iterator in [domain.name for domain in getDomainsWithParallelRegionTemplate(template)]
        ))
    iterators = iteratorsByName.keys()
    if len(iterators) == 0:
        return result
    result += "integer(4) :: "
    for index, iterator in enumerate(iterators):
        if index != 0:
            result = result + ", "
        result = result + iterator
    result += "\n"
    return result

def getErrorHandlingAfterKernelCall(calleeRoutineNode, errorVariable="cuerror", stopImmediately=True):
    name = calleeRoutineNode.getAttribute('name')
    if not name:
        raise Exception("Unexpected Error: routine node without name")
    stopLine = ""
    if stopImmediately:
        stopLine = "stop 1\n"
    return  "%s = cudaThreadSynchronize()\n" \
            "%s = cudaGetLastError()\n" \
            "if(%s .NE. cudaSuccess) then\n"\
                "\twrite(0, *) 'CUDA error in kernel %s:', cudaGetErrorString(%s)\n%s" \
            "end if\n" %(errorVariable, errorVariable, errorVariable, name, errorVariable, stopLine)

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

def getRuntimeDebugPrintStatements(symbolsByName, calleeRoutineNode, parallelRegionNode):
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
        result = result + "cuTemp = %s\n" %(symbol.accessRepresentation(iterators, offsets, parallelRegionNode))
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
    onDevice = False
    multipleParallelRegionsPerSubroutineAllowed = True
    optionFlags = []

    def __init__(self, optionFlags):
        self.currDependantSymbols = None
        if type(optionFlags) == list:
            self.optionFlags = optionFlags

    def filePreparation(self, filename):
        return '''#include "storage_order.F90"\n'''

    def warningOnUnrecognizedSubroutineCallInParallelRegion(self, callerName, calleeName):
        return ""

    def kernelCallConfig(self):
        return ""

    def kernelCallPreparation(self, parallelRegionTemplate, calleeNode=None):
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
        return [domain.name for domain in getDomainsWithParallelRegionTemplate(parallelRegionTemplate)]

    def iteratorDefinitionBeforeParallelRegion(self, domains):
        return ""

    def safetyOutsideRegion(self, domains):
        return ""

    def loopPreparation(self):
        return ""

    def parallelRegionBegin(self, parallelRegionTemplate):
        domains = getDomainsWithParallelRegionTemplate(parallelRegionTemplate)
        regionStr = ''
        for pos in range(len(domains)-1,-1,-1): #use inverted order (optimization of accesses for fortran storage order)
            domain = domains[pos]
            startsAt = domain.startsAt if domain.startsAt != None else "1"
            endsAt = domain.endsAt if domain.endsAt != None else domain.size
            regionStr = regionStr + 'do %s=%s,%s' %(domain.name, startsAt, endsAt)
            if pos != 0:
                regionStr = regionStr + '\n '
            pos = pos + 1
        return regionStr

    def parallelRegionEnd(self, parallelRegionTemplate):
        domains = getDomainsWithParallelRegionTemplate(parallelRegionTemplate)
        pos = 0
        regionStr = ''
        for domain in domains:
            regionStr = regionStr + 'end do'
            if pos != len(domains) - 1:
                regionStr = regionStr + '\n '
            pos = pos + 1
        return regionStr

    def declarationEnd(self, dependantSymbols, routineIsKernelCaller, currRoutineNode, currParallelRegionTemplates):
        return getIteratorDeclaration(currRoutineNode, currParallelRegionTemplates, ["CPU", ""])

    def subroutineEnd(self, dependantSymbols, routineIsKernelCaller):
        return ''

class OpenMPFortranImplementation(FortranImplementation):
    def __init__(self, optionFlags):
        self.currDependantSymbols = None
        FortranImplementation.__init__(self, optionFlags)

    def declarationEnd(self, dependantSymbols, routineIsKernelCaller, currRoutineNode, currParallelRegionTemplates):
        self.currDependantSymbols = dependantSymbols
        return FortranImplementation.declarationEnd(self, dependantSymbols, routineIsKernelCaller, currRoutineNode, currParallelRegionTemplates)

    def parallelRegionBegin(self, parallelRegionTemplate):
        if self.currDependantSymbols == None:
            raise Exception("parallel region without any dependant arrays")
        openMPLines = "!$OMP PARALLEL DO DEFAULT(Private) "
        openMPLines += "SHARED(%s)\n" %(', '.join([symbol.deviceName() for symbol in self.currDependantSymbols]))
        return openMPLines + FortranImplementation.parallelRegionBegin(self, parallelRegionTemplate)

    def parallelRegionEnd(self, parallelRegionTemplate):
        openMPLines = "\n!$OMP END PARALLEL DO\n"
        return FortranImplementation.parallelRegionEnd(self, parallelRegionTemplate) + openMPLines

    def subroutineEnd(self, dependantSymbols, routineIsKernelCaller):
        self.currDependantSymbols = None
        return FortranImplementation.subroutineEnd(self, dependantSymbols, routineIsKernelCaller)

class PGIOpenACCFortranImplementation(FortranImplementation):
    onDevice = True

    def filePreparation(self, filename):
        additionalStatements = '''
attributes(global) subroutine HF_DUMMYKERNEL_%s()
use cudafor
!This ugly hack is used because otherwise as of PGI 14.7, OpenACC kernels could not be used in code that is compiled with CUDA flags.
end subroutine
        ''' %(os.path.basename(filename).split('.')[0])
        return FortranImplementation.filePreparation(self, filename) + additionalStatements

    def declarationEnd(self, dependantSymbols, routineIsKernelCaller, currRoutineNode, currParallelRegionTemplates):
        result = ""
        result += getIteratorDeclaration(currRoutineNode, currParallelRegionTemplates, ["GPU"])
        dataDeclarations = ""
        dataDeclarations += "!$acc declare "
        dataDeclarationsRequired = False
        commaRequired = False
        for index, symbol in enumerate(dependantSymbols):
            #Rules that lead to a symbol not being touched by directives
            if not symbol.domains or len(symbol.domains) == 0:
                continue

            #Rules for symbols that are declared present
            newDataDeclarations = ""
            if symbol.isPresent:
                newDataDeclarations += "present(%s)" %(symbol.name)

            #Rules for kernel wrapper routines and symbols declared to be transfered
            elif (symbol.intent == "in") \
            and (routineIsKernelCaller or symbol.isToBeTransfered):
                newDataDeclarations += "copyin(%s)" %(symbol.name)
            elif (symbol.intent == "inout" or not symbol.sourceModule in [None,""] or symbol.isHostSymbol) \
            and (routineIsKernelCaller or symbol.isToBeTransfered):
                newDataDeclarations += "copy(%s)" %(symbol.name)
            elif (symbol.intent == "out") \
            and (routineIsKernelCaller or symbol.isToBeTransfered):
                newDataDeclarations += "copyout(%s)" %(symbol.name)

            #Rules for other routines
            elif not routineIsKernelCaller \
            and currRoutineNode.getAttribute('parallelRegionPosition') != 'within':
                continue

            #Rules for kernels
            elif currRoutineNode.getAttribute('parallelRegionPosition') == 'within' \
            and (symbol.intent in ["in", "inout", "out"] or not symbol.sourceModule in [None,""] or symbol.isHostSymbol):
                newDataDeclarations += "present(%s)" %(symbol.name)
            else:
                newDataDeclarations += "create(%s)" %(symbol.name)

            #Wrapping up
            if commaRequired == True:
                newDataDeclarations = ", " + newDataDeclarations
            dataDeclarations += newDataDeclarations
            dataDeclarationsRequired = True
            commaRequired = True
        dataDeclarations += "\n"
        if dataDeclarationsRequired == True:
            result += dataDeclarations
        return result

    def getIterators(self, parallelRegionTemplate):
        if not appliesTo(["GPU"], parallelRegionTemplate):
            return []
        return [domain.name for domain in getDomainsWithParallelRegionTemplate(parallelRegionTemplate)]

    def loopPreparation(self):
        return "!$acc loop seq"

    def parallelRegionBegin(self, parallelRegionTemplate):
        vectorSizePPNames = getVectorSizePPNames(parallelRegionTemplate)
        regionStr = "!$acc kernels\n"
        domains = getDomainsWithParallelRegionTemplate(parallelRegionTemplate)
        if len(domains) > 3 or len(domains) < 1:
            raise Exception("Invalid number of parallel domains in parallel region definition.")
        for pos in range(len(domains)-1,-1,-1): #use inverted order (optimization of accesses for fortran storage order)
            regionStr += "!$acc loop independent vector(%s)\n" %(vectorSizePPNames[pos])
            domain = domains[pos]
            startsAt = domain.startsAt if domain.startsAt != None else "1"
            endsAt = domain.endsAt if domain.endsAt != None else domain.size
            regionStr += 'do %s=%s,%s' %(domain.name, startsAt, endsAt)
            if pos != 0:
                regionStr += '\n '
            pos = pos + 1
        return regionStr

    def parallelRegionEnd(self, parallelRegionTemplate):
        additionalStatements = "\n!$acc end kernels\n"
        return FortranImplementation.parallelRegionEnd(self, parallelRegionTemplate) + additionalStatements

class CUDAFortranImplementation(FortranImplementation):
    onDevice = True
    multipleParallelRegionsPerSubroutineAllowed = False

    def __init__(self, optionFlags):
        self.currRoutineNode = None
        self.currParallelRegionTemplateNode = None
        FortranImplementation.__init__(self, optionFlags)

    def warningOnUnrecognizedSubroutineCallInParallelRegion(self, callerName, calleeName):
        return "WARNING: subroutine %s called inside %s's parallel region, but it is not defined in a h90 file.\n" \
                    %(calleeName, callerName)

    def kernelCallConfig(self):
        return "<<< cugrid, cublock >>>"

    def kernelCallPreparation(self, parallelRegionTemplate, calleeNode=None):
        if not appliesTo(["GPU"], parallelRegionTemplate):
            return ""
        self.currParallelRegionTemplateNode = parallelRegionTemplate
        gridPreparationStr = ""
        if calleeNode != None and "DO_NOT_TOUCH_GPU_CACHE_SETTINGS" not in self.optionFlags:
            gridPreparationStr += "cuerror = cudaFuncSetCacheConfig(%s, cudaFuncCachePreferL1)\n" %(calleeNode.getAttribute('name'))
            gridPreparationStr += "cuerror = cudaGetLastError()\n"
            gridPreparationStr += "if(cuerror .NE. cudaSuccess) then\n \
    \twrite(0, *) 'CUDA error when setting cache configuration for kernel %s:', cudaGetErrorString(cuerror)\n \
    stop 1\n\
end if\n" %(calleeNode.getAttribute('name'))
        blockSizePPNames = getVectorSizePPNames(parallelRegionTemplate)
        gridSizeVarNames = ["cugridSizeX", "cugridSizeY", "cugridSizeZ"]
        domains = getDomainsWithParallelRegionTemplate(parallelRegionTemplate)
        if len(domains) > 3 or len(domains) < 1:
            raise Exception("Invalid number of parallel domains in parallel region definition.")
        blockStr = "cublock = dim3("
        gridStr = "cugrid = dim3("
        for i in range(3):
            if i != 0:
                gridPreparationStr += "\n"
                gridStr += ", "
                blockStr += ", "
            if i < len(domains):
                gridPreparationStr += "%s = ceiling(real(%s) / real(%s))" %(gridSizeVarNames[i], domains[i].size, blockSizePPNames[i])
                blockStr += "%s" %(blockSizePPNames[i])
            else:
                gridPreparationStr +=  "%s = 1" %(gridSizeVarNames[i])
                blockStr += "1"
            gridStr += "%s" %(gridSizeVarNames[i])
        result = gridPreparationStr + "\n" + gridStr + ")\n" + blockStr + ")\n"
        return result

    def kernelCallPost(self, symbolsByName, calleeRoutineNode):
        self.currParallelRegionTemplateNode = None
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
        return [domain.name for domain in getDomainsWithParallelRegionTemplate(parallelRegionTemplate)]

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
        self.currRoutineNode = None
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
                if symbol.isPointer:
                    deviceInitStatements += "deallocate(%s)\n" %(symbol.deviceName())
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

    def iteratorDefinitionBeforeParallelRegion(self, domains):
        if len(domains) > 3:
            raise Exception("Only up to 3 parallel dimensions supported. %i are specified: %s." %(len(domains), str(domains)))
        cudaDims = ("x", "y", "z")
        result = ""
        for index, domain in enumerate(domains):
            result += "%s = (blockidx%%%s - 1) * blockDim%%%s + threadidx%%%s\n" %(domain.name, cudaDims[index], cudaDims[index], cudaDims[index])
        return result

    def safetyOutsideRegion(self, domains):
        result = "if ("
        for index, domain in enumerate(domains):
            startsAt = domain.startsAt if domain.startsAt != None else "1"
            endsAt = domain.endsAt if domain.endsAt != None else domain.size
            if index != 0:
                result += " .OR. "
            result += "%s .GT. %s .OR. %s .LT. %s" %(domain.name, endsAt, domain.name, startsAt)
        result += ") then\nreturn\nend if\n"
        return result

    def parallelRegionBegin(self, parallelRegionTemplate):
        self.currParallelRegionTemplateNode = parallelRegionTemplate
        domains = getDomainsWithParallelRegionTemplate(parallelRegionTemplate)
        regionStr = self.iteratorDefinitionBeforeParallelRegion(domains)
        regionStr += self.safetyOutsideRegion(domains)
        return regionStr

    def parallelRegionEnd(self, parallelRegionTemplate):
        self.currParallelRegionTemplateNode = None
        return ""

    def declarationEnd(self, dependantSymbols, routineIsKernelCaller, currRoutineNode, currParallelRegionTemplates):
        self.currRoutineNode = currRoutineNode
        result = ""
        result += getIteratorDeclaration(currRoutineNode, currParallelRegionTemplates, ["GPU"])

        if routineIsKernelCaller:
            result = result + "type(dim3) :: cugrid, cublock\n"
            result = result + "integer(4) :: cugridSizeX, cugridSizeY, cugridSizeZ, cuerror, cuErrorMemcopy\n"

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
                if symbol.isPointer:
                    deviceInitStatements += "allocate(%s)\n" %(symbol.allocationRepresentation())
                deviceInitStatements += deviceStr + " = " + originalStr + "\n"
            elif (routineIsKernelCaller or symbol.isToBeTransfered):
                deviceInitStatements += symbol.selectAllRepresentation() + " = 0\n"

        return result + deviceInitStatements

    def additionalIncludes(self):
        return "use cudafor\n"

class DebugCUDAFortranImplementation(CUDAFortranImplementation):

    def declarationEnd(self, dependantSymbols, routineIsKernelCaller, currRoutineNode, currParallelRegionTemplates):
        result = "real(8) :: cuTemp\n"
        result = result + CUDAFortranImplementation.declarationEnd(self, dependantSymbols, routineIsKernelCaller, \
            currRoutineNode, currParallelRegionTemplates)
        return result

    def kernelCallPreparation(self, parallelRegionTemplate, calleeNode=None):
        result = CUDAFortranImplementation.kernelCallPreparation(self, parallelRegionTemplate, calleeNode)
        iterators = self.getIterators(parallelRegionTemplate)
        gridSizeVarNames = ["cugridSizeX", "cugridSizeY", "cugridSizeZ"]
        routineName = self.currRoutineNode.getAttribute('name')
        result += "write(0,*) 'calling kernel %s with grid size', " %(routineName)
        for i in range(len(iterators)):
            if i != 0:
                result += ", "
            result += "%s" %(gridSizeVarNames[i])
        result += "\n"
        return result

    def kernelCallPost(self, symbolsByName, calleeRoutineNode):
        if calleeRoutineNode.getAttribute('parallelRegionPosition') != 'within':
            return ""
        result = getErrorHandlingAfterKernelCall(calleeRoutineNode, errorVariable="cuerror", stopImmediately=False)
        #michel 2013-4-18: Note, that doing the stop *after* trying to print the memory state hasn't really helped so far.
        #CUDA seems to fail any memcopy attempts after a kernel fails - maybe there is some method to clear the error state before
        #doing memcopy?
        result = result + getRuntimeDebugPrintStatements(symbolsByName, calleeRoutineNode, self.currParallelRegionTemplateNode)
        result = result + "if(cuerror .NE. cudaSuccess) then\n"\
                "\tstop 1\n" \
            "end if\n"
        #TODO: remove
        #result = result + getTempDeallocationsAfterKernelCall(symbolsByName)
        return result

class DebugEmulatedCUDAFortranImplementation(DebugCUDAFortranImplementation):

    def __init__(self, optionFlags):
        self.currDependantSymbols = None
        DebugCUDAFortranImplementation.__init__(self, optionFlags)

    def declarationEnd(self, dependantSymbols, routineIsKernelCaller, currRoutineNode, currParallelRegionTemplates):
        self.currDependantSymbols = dependantSymbols
        return DebugCUDAFortranImplementation.declarationEnd(self, dependantSymbols, routineIsKernelCaller, currRoutineNode, currParallelRegionTemplates)

    def parallelRegionBegin(self, parallelRegionTemplate):
        domains = getDomainsWithParallelRegionTemplate(parallelRegionTemplate)
        regionStr = self.iteratorDefinitionBeforeParallelRegion(domains)
        routineName = self.currRoutineNode.getAttribute('name')
        if not routineName:
            raise Exception("Routine name undefined.")
        iterators = self.getIterators(parallelRegionTemplate)
        if not iterators or len(iterators) == 0:
            raise Exception("No iterators in kernel.")
        error_conditional = "("
        for i in range(len(iterators)):
            if i != 0:
                error_conditional += " .OR. "
            error_conditional += "%s .LT. 1" %(iterators[i])
        error_conditional = error_conditional + ")"
        regionStr += "if %s then\n\twrite(0,*) 'ERROR: invalid initialization of iterators in kernel \
%s - check kernel domain setup'\n" %(error_conditional, routineName)
        regionStr += "\twrite(0,*) "
        for i in range(len(iterators)):
            if i != 0:
                regionStr += ", "
            regionStr += "'%s', %s" %(iterators[i], iterators[i])
        regionStr += "\nend if\n"
        conditional = "("
        for i in range(len(iterators)):
            if i != 0:
                conditional = conditional + " .AND. "
            conditional = conditional + "%s .EQ. %i" %(iterators[i], 1)
        conditional = conditional + ")"
        regionStr += "if %s write(0,*) '*********** entering kernel %s finished *************** '\n" %(conditional, routineName)
        region_domains = getDomainsWithParallelRegionTemplate(parallelRegionTemplate)
        for symbol in self.currDependantSymbols:
            offsets = []
            symbol_domain_names = [domain[0] for domain in symbol.domains]
            for region_domain in region_domains:
                if region_domain.name in symbol_domain_names:
                    offsets.append(region_domain.startsAt if region_domain.startsAt != None else "1")
            for i in range(len(symbol.domains) - len(offsets)):
                offsets.append("1")
            if symbol.intent == "in" or symbol.intent == "inout":
                symbol_access = symbol.accessRepresentation(iterators, offsets, parallelRegionTemplate)
                regionStr += "if %s then\n\twrite(0,*) '%s', %s\nend if\n" %(conditional, symbol_access, symbol_access)

        regionStr += "if %s write(0,*) '**********************************************'\n" %(conditional)
        regionStr += "if %s write(0,*) ''\n" %(conditional)
        regionStr += self.safetyOutsideRegion(domains)
        return regionStr

    def subroutineEnd(self, dependantSymbols, routineIsKernelCaller):
        self.currDependantSymbols = None
        return DebugCUDAFortranImplementation.subroutineEnd(self, dependantSymbols, routineIsKernelCaller)




