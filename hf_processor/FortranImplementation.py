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
from H90Symbol import Symbol, DeclarationType, purgeFromDeclarationSettings
from H90RegExPatterns import H90RegExPatterns
from DomHelper import *
import os
import sys
import re

number_of_traces = 200

def getReductionClause(parallelRegionTemplate):
    reductionScalarsByOperator = getReductionScalarsByOperator(parallelRegionTemplate)
    return ", ".join([
        "reduction(%s: %s)" %(operator, ", ".join(reductionScalarsByOperator[operator]))
        for operator in reductionScalarsByOperator.keys()
    ])

def getDataDirectiveAndUpdateOnDeviceFlags(currRoutineNode, currParallelRegionTemplates, dependantSymbols, createDeclaration, routineIsKernelCaller, debugPrint, enterOrExit='enter'):
    presentDeclaration = "present" # if currRoutineNode.getAttribute("parallelRegionPosition") == 'inside' else "deviceptr"
    copyDeclaration = "copyin"
    if enterOrExit != 'enter':
        copyDeclaration = "copyout"
    result = ""
    dataDeclarations = ""
    if enterOrExit == 'enter':
        dataDeclarations += "!$acc enter data "
    else:
        dataDeclarations += "!$acc exit data "
    dataDeclarationsRequired = False
    commaRequired = False
    for index, symbol in enumerate(dependantSymbols):
        if debugPrint:
            sys.stderr.write(
                "analyzing symbol %s for data directive. Domains: %s, IsHostSymbol: %s, IsPresent: %s, IsToBeTransfered: %s, SourceModule: %s, Intent: %s\n" %(
                    symbol.name,
                    str(symbol.domains),
                    symbol.isHostSymbol,
                    symbol.isPresent,
                    symbol.isToBeTransfered,
                    str(symbol.sourceModule),
                    str(symbol.intent)
                )
            )
        #Rules that lead to a symbol not being touched by directives
        symbol.isOnDevice = False
        if not symbol.domains or len(symbol.domains) == 0:
            continue
        if symbol.isHostSymbol:
            continue
        if currRoutineNode.getAttribute('parallelRegionPosition') == 'within'\
        and (symbol.intent in ["in", "inout", "out", "unspecified"] or not symbol.sourceModule in [None,""]):
            symbol.isOnDevice = True
            continue
        if currRoutineNode.getAttribute('parallelRegionPosition') != 'inside'\
        and not symbol.sourceModule in [None,""]:
            continue

        #Rules for kernel wrapper routines and symbols declared to be transfered
        newDataDeclarations = ""
        if symbol.isPresent:
            if symbol.intent in ["in", "out", "inout", "unspecified"] or not symbol.sourceModule in [None,""]:
                #all we can do is marking the symbol correctly - OpenACC enter data doesn't support present check sadly
                #please note: unspecified intent is a symbol that is a dummy variable with no intent specified.
                if enterOrExit == 'enter':
                    symbol.isOnDevice = True
                else:
                    symbol.isOnDevice = False
                continue
            else:
                #no intent, no source module specified --> local variable
                if enterOrExit == 'enter':
                    newDataDeclarations += "%s(%s)" %(createDeclaration, symbol.name)
                    symbol.isOnDevice = True
                else:
                    newDataDeclarations += "delete(%s)" %(symbol.name)
                    symbol.isOnDevice = False
        else:
            if not routineIsKernelCaller and currRoutineNode.getAttribute('parallelRegionPosition') != 'within' and not symbol.isToBeTransfered:
                continue
            if symbol.intent == "in":
                if enterOrExit == 'enter':
                    newDataDeclarations += "copyin(%s)" %(symbol.name)
                    symbol.isOnDevice = True
                else:
                    newDataDeclarations += "delete(%s)" %(symbol.name)
                    symbol.isOnDevice = False
            elif symbol.intent == "inout" or not symbol.sourceModule in [None,""]:
                newDataDeclarations += "%s(%s)" %(copyDeclaration, symbol.name)
                if enterOrExit == 'enter':
                    symbol.isOnDevice = True
                else:
                    symbol.isOnDevice = True
            elif symbol.intent == "out":
                #We need to be careful here: Because of branching around kernels it could easily happen that
                #copyout data is not being written inside the data region, thus overwriting the host data with garbage.
                newDataDeclarations += "%s(%s)" %(copyDeclaration, symbol.name)
                if enterOrExit == 'enter':
                    symbol.isOnDevice = True
                else:
                    symbol.isOnDevice = False
            elif enterOrExit == 'enter':
                newDataDeclarations += "%s(%s)" %(createDeclaration, symbol.name)
                symbol.isOnDevice = True
            else:
                newDataDeclarations += "delete(%s)" %(symbol.name)
                symbol.isOnDevice = False

        #Wrapping up enter data / exit data
        if commaRequired == True:
            newDataDeclarations = ", " + newDataDeclarations
        dataDeclarations += newDataDeclarations
        dataDeclarationsRequired = True
        commaRequired = True

    dataDeclarations += "\n"
    if dataDeclarationsRequired == True:
        result += dataDeclarations
    return result, dataDeclarationsRequired

def getLoopOverSymbolValues(symbol, loopName, innerLoopImplementationFunc):
    result = ""
    if len(symbol.domains) > 0:
        result += "hf_tracing_outer_%s: " %(loopName)
    for domainNum in range(len(symbol.domains),0,-1):
        result += "do hf_tracing_enum%i = lbound(hf_tracing_temp_%s,%i), ubound(hf_tracing_temp_%s,%i)\n" %(domainNum, symbol.name, domainNum, symbol.name, domainNum)
    result += innerLoopImplementationFunc(symbol)
    for domainNum in range(len(symbol.domains),0,-1):
        result += "end do"
        if domainNum != 1:
            result+= '\n'
    if len(symbol.domains) > 0:
        result += " hf_tracing_outer_%s\n" %(loopName)
    return result

def getTracingDeclarationStatements(currRoutineNode, dependantSymbols, patterns, useReorderingByAdditionalSymbolPrefixes={'hf_tracing_temp_':False}):
    tracing_symbols = []
    if len(dependantSymbols) == 0 or currRoutineNode.getAttribute('parallelRegionPosition') == 'outside':
        return "", tracing_symbols

    result = "integer(8), save :: hf_tracing_counter = 0\n"
    result += "integer(4) :: hf_error_printed_counter\n"
    result += "character(len=256) :: hf_tracing_current_path\n"
    max_num_of_domains_for_symbols = 0
    for symbol in dependantSymbols:
        if len(symbol.domains) == 0:
        # or 'allocatable' in symbol.declarationPrefix \
        # or symbol.intent not in ['in', 'inout', 'out'] \
        # or symbol.isOnDevice and currRoutineNode.getAttribute('parallelRegionPosition') == 'inside':
            continue
        if len(symbol.domains) > max_num_of_domains_for_symbols:
            max_num_of_domains_for_symbols = len(symbol.domains)
        for prefix in useReorderingByAdditionalSymbolPrefixes.keys():
            current_declaration_line = symbol.getDeclarationLineForAutomaticSymbol(
                purgeList=['intent', 'public', 'allocatable', 'target'],
                patterns=patterns,
                name_prefix=prefix,
                use_domain_reordering=useReorderingByAdditionalSymbolPrefixes[prefix],
                skip_on_missing_declaration=True
            )
            if current_declaration_line == "":
                break
            result += current_declaration_line + '\n'
        else:
            tracing_symbols.append(symbol)

    if max_num_of_domains_for_symbols > 0:
        result += "integer(4) :: %s\n" %(
            ', '.join(
                ["hf_tracing_enum%i" %(domainNum) for domainNum in range(1,max_num_of_domains_for_symbols+1)]
            )
        )
    return result, tracing_symbols

def getTracingStatements(currRoutineNode, currModuleName, tracingSymbols, traceHandlingFunc, increment_tracing_counter=True, loop_name_postfix=''):
    def innerTempArraySetterLoopFunc(symbol):
        return "hf_tracing_temp_%s = %s\n" %(
            symbol.accessRepresentation(
                parallelIterators=[],
                offsets=["hf_tracing_enum%i" %(domainNum) for domainNum in range(1,len(symbol.domains)+1)],
                parallelRegionNode=None,
                use_domain_reordering=False
            ),
            symbol.accessRepresentation(
                parallelIterators=[],
                offsets=["hf_tracing_enum%i" %(domainNum) for domainNum in range(1,len(symbol.domains)+1)],
                parallelRegionNode=None,
                use_domain_reordering=True
            )
        )

    result = ''
    if len(tracingSymbols) > 0 and currRoutineNode.getAttribute('parallelRegionPosition') != 'outside':
        result += "if (hf_tracing_counter .lt. %i) then\n" %(number_of_traces)
        for symbol in tracingSymbols:
            if 'allocatable' in symbol.declarationPrefix:
                result += "if (allocated(%s)) then\n" %(symbol.name)
            result += "hf_tracing_temp_%s = 0\n" %(
                symbol.accessRepresentation(
                    parallelIterators=[],
                    offsets=[":" for _ in range(len(symbol.domains))],
                    parallelRegionNode=None,
                    use_domain_reordering=False
                )
            )
            if symbol.isOnDevice:
                result += "!$acc update host(%s) if(hf_symbols_are_device_present)\n" %(symbol.name)
            result += getLoopOverSymbolValues(symbol, "%s_temp_%s" %(symbol.name, loop_name_postfix), innerTempArraySetterLoopFunc)
            result += traceHandlingFunc(currRoutineNode, currModuleName, symbol)
            if 'allocatable' in symbol.declarationPrefix:
                result += "end if\n"
        result += "end if\n"
    if increment_tracing_counter:
        result += "hf_tracing_counter = hf_tracing_counter + 1\n"
    return result

def tracingFilename(currModuleName, currRoutineNode, symbol, begin_or_end):
    filename_postfix = "%s_" %(begin_or_end)
    if symbol.intent == "inout" and begin_or_end == "begin":
        filename_postfix = "in_"
    elif symbol.intent == "inout":
        filename_postfix = "out_"
    elif symbol.intent not in ["", None]:
        filename_postfix = "%s_" %(symbol.intent)
    return "%s_%s_%s_%s" %(
        currModuleName,
        currRoutineNode.getAttribute('name'),
        symbol.name,
        filename_postfix
    )

def getCompareToTraceFunc(abortSubroutineOnError=True, loop_name_postfix='', begin_or_end='begin'):
    def printSomeErrors(symbol):
        accessor = symbol.accessRepresentation(
            parallelIterators=[],
            offsets=["hf_tracing_enum%i" %(domainNum) for domainNum in range(1,len(symbol.domains)+1)],
            parallelRegionNode=None,
            use_domain_reordering=False
        )
        result = ""
        result += "if ( ( abs( hf_tracing_comparison_%s - hf_tracing_temp_%s ) / hf_tracing_comparison_%s ) .gt. 1E-9) then \n" %(accessor, accessor, accessor)
        result += "write(0,*) 'error at:'\n"
        for domainNum in range(1,len(symbol.domains)+1):
            result += "write(0,*) 'domain %i:', hf_tracing_enum%i\n" %(domainNum, domainNum)
        result += "write(0,*) 'expected', hf_tracing_comparison_%s, 'actual', hf_tracing_temp_%s\n" %(accessor, accessor)
        result += "hf_error_printed_counter = hf_error_printed_counter + 1\n"
        result += "if (hf_error_printed_counter >= 5) then\n"
        result += "exit hf_tracing_outer_%s_printindex_%s\n" %(symbol.name, loop_name_postfix)
        result += "end if\n"
        result += "end if\n"
        return result

    def compareToTrace(currRoutineNode, currModuleName, symbol):
        result = "call findNewFileHandle(hf_tracing_imt)\n"
        result += "write(hf_tracing_current_path, '(A,I3.3,A)') './datatrace/%s', hf_tracing_counter, '.dat'\n" %(
            tracingFilename(currModuleName, currRoutineNode, symbol, begin_or_end)
        )
        result += "open(hf_tracing_imt, file=trim(hf_tracing_current_path), form='unformatted', status='old', action='read', iostat=hf_tracing_ierr)\n"
        result += "if (hf_tracing_ierr .ne. 0) then\n"
        result += "write(0,*) 'symbol %s trace n/a'\n" %(symbol.name)
        result += "close(hf_tracing_imt)\n"
        if abortSubroutineOnError:
            result += "hf_tracing_counter = hf_tracing_counter + 1\n"
            result += "return\n"
        result += "else\n"
        result += "read(hf_tracing_imt) hf_tracing_comparison_%s\n" %(symbol.name)
        result += "hf_num_of_elements = " + ' * '.join([
            "(ubound(hf_tracing_comparison_%s,%i) - lbound(hf_tracing_comparison_%s,%i) + 1)" %(symbol.name, domainNum, symbol.name, domainNum)
            for domainNum in range(1,len(symbol.domains)+1)
        ]) + '\n'
        result += "if (hf_num_of_elements <= 0) then\n"
        result += "write(0,*) '%s: symbol %s ok. (array has no elements)'\n" %(currRoutineNode.getAttribute('name'), symbol.name)
        result += "else\n"
        if 'real' in symbol.declarationPrefix:
            result += "hf_mean_ref = sqrt(sum(hf_tracing_comparison_%s**2) / hf_num_of_elements)\n" %(symbol.name)
            result += "hf_mean_gpu = sqrt(sum(hf_tracing_temp_%s**2) / hf_num_of_elements)\n" %(symbol.name)
            result += "if (abs(hf_mean_ref) .gt. 1E-20) then\n"
            result += "hf_tracing_error = sqrt(sum((hf_tracing_comparison_%s - hf_tracing_temp_%s)**2 ) / hf_num_of_elements ) / hf_mean_ref\n" %(
                symbol.name,
                symbol.name
            )
            result += "else\n"
            result += "hf_tracing_error = sqrt(sum((hf_tracing_comparison_%s - hf_tracing_temp_%s)**2 ) / hf_num_of_elements )\n" %(
                symbol.name,
                symbol.name
            )
            result += "end if\n"
            result += "if (hf_tracing_error .ne. hf_tracing_error .or. hf_tracing_error .gt. 1E-9) then\n" # a .ne. a tests a for NaN in Fortran (needs -Kieee compiler flag in pgf90)
            result += "\
write(0,*) 'In module %s, subroutine %s, checkpoint %s:', 'Real Array %s does not match the data found in ', trim(hf_tracing_current_path), \
' RMS Error:', hf_tracing_error, ' NumOfValues:', hf_num_of_elements, ' ReferenceMean: ', hf_mean_ref, ' GPUMean: ', hf_mean_gpu\n\
            " %(
                currModuleName,
                currRoutineNode.getAttribute('name'),
                loop_name_postfix,
                symbol.name
            )
            result += "hf_error_printed_counter = 0\n"
            result += getLoopOverSymbolValues(symbol, "%s_printindex_%s" %(symbol.name, loop_name_postfix), printSomeErrors)
        else:
            result += "if (any(hf_tracing_comparison_%s .ne. hf_tracing_temp_%s)) then\n" %(symbol.name, symbol.name)
            result += "write(0,*) 'In module %s, subroutine %s, checkpoint %s::', 'Array %s does not match the data found in ./datatrace.'\n" %(
                currModuleName,
                currRoutineNode.getAttribute('name'),
                loop_name_postfix,
                symbol.name
            )
        result += "write(0,*) 'GPU version shape:'\n"
        result += "write(0,*) shape(hf_tracing_temp_%s)\n" %(symbol.name)
        result += "write(0,*) 'Reference version shape:'\n"
        result += "write(0,*) shape(hf_tracing_comparison_%s)\n" %(symbol.name)
        result += "hf_tracing_error_found = .true.\n"
        result += "else\n"
        result += "write(0,*) 'symbol %s ok. nRMSE: ', hf_tracing_error\n" %(symbol.name)
        result += "end if\n"
        result += "end if\n"
        result += "end if\n"
        result += "close(hf_tracing_imt)\n"
        return result
    return compareToTrace

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

def getCUDAErrorHandling(calleeRoutineNode, errorVariable="cuerror", stopImmediately=True):
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

def getDebugOffsetString(domainTuple, previousOffsets):
    def getUpperBound(domainSizeSpec):
        boundaries = domainSizeSpec.split(':')
        if len(boundaries) == 1:
            return boundaries[0].strip()
        if len(boundaries) == 2:
            return boundaries[1].strip()
        raise Exception("Unexpected domain size specification: %s" %(domainSizeSpec))

    #$$$ change this - it must be consistant with storage_order.F90
    userdefinedDomNames = ["x", "y", "z", "nz", "i", "j", "vertical", "verticalPlus", "KMAX_CONST", "KMP1_CONST", "ntlm", "ngm", "id_qa_e", "1", "2", "3", "4"]
    (dependantDomName, dependantDomSize) = domainTuple
    upperBound = getUpperBound(dependantDomSize)
    offset = ""
    if dependantDomName in userdefinedDomNames:
        offset = "DEBUG_OUT_%s" %(dependantDomName.strip())
    elif upperBound in userdefinedDomNames:
        offset = "DEBUG_OUT_%s" %(upperBound)
    else:
        offset = "DEBUG_OUT_x"
    if offset in previousOffsets:
        #if there are multiple dimensions with the same sizes, use the second specified macro
        # - else we'd be always showing the diagonal of quadratic matrices.
        offset += "_2"
    return offset

def getRuntimeDebugPrintStatements(symbolsByName, calleeRoutineNode, parallelRegionNode, useOpenACC=True):
    def wrap_in_acc_pp(string, symbol):
        accPP = symbol.accPP()[0]
        if accPP == "":
            return string
        return accPP + "(" + string + ")"

    result = ""
    if calleeRoutineNode.getAttribute('parallelRegionPosition') == 'outside':
        result += "#ifndef GPU\n"
        result += "if (hf_debug_print_iterator == 0) then\n"

    routineName = calleeRoutineNode.getAttribute('name')
    if not routineName:
        raise Exception("Callee routine name undefined.")
    result += "write(0,*) '*********** kernel %s finished *************** '\n" %(routineName)
    symbolsToPrint = symbolsByName.values()
    # symbolNames = sorted(symbolsByName.keys())
    # symbolsToPrint = [
    #     symbolsByName[symbolName] for symbolName in symbolNames
    #     if symbolsByName[symbolName].domains and len(symbolsByName[symbolName].domains) != 0
    # ]
    offsetsBySymbolName = {}
    for symbol in symbolsToPrint:
        offsets = []
        for domain in symbol.domains:
            offsets.append(getDebugOffsetString(domain, offsets))
        offsetsBySymbolName[symbol.name] = offsets
    symbolClauses = [
        symbol.accessRepresentation(
            [],
            ["%s:%s" %(offset, offset) for offset in offsetsBySymbolName[symbol.name]],
            parallelRegionNode
        )
        for symbol in symbolsToPrint
        if len(symbol.domains) > 0
    ]
    if useOpenACC:
        result += "#ifdef GPU\n"
        result += "!$acc update if(hf_symbols_are_device_present) host(%s)\n" %(", ".join(symbolClauses)) if len(symbolsToPrint) > 0 else ""
        result += "#endif\n"
    for symbol in symbolsToPrint:
        result = result + "hf_output_temp = %s\n" %(symbol.accessRepresentation([], offsetsBySymbolName[symbol.name], parallelRegionNode))
        #michel 2013-4-18: the Fortran-style memcopy as used right now in the above line creates a runtime error immediately
        #                  if we'd like to catch such errors ourselves, we need to use the cuda API memcopy calls - however we
        #                  then also need information about the symbol size, which isn't available in the current implementation
        #                  (we currently look at the typedef just as a string).
        # result = result + "cuErrorMemcopy = cudaGetLastError()\n" \
        #     "if(cuErrorMemcopy .NE. cudaSuccess) then\n"\
        #         "\twrite(0, *) 'CUDA error when attempting to copy value from %s:', cudaGetErrorString(cuErrorMemcopy)\n" \
        #         "stop 1\n" \
        #     "end if\n" %(symbol.name)
        joinedDomains = offsetsBySymbolName[symbol.name]
        if len(joinedDomains) > 0:
            domainsStr = "(',"
            formStr = "'(A,"
            for i in range(len(joinedDomains)):
                if i != 0:
                    domainsStr = domainsStr + ",', ',"
                    formStr = formStr + ",A,"
                domainsStr = domainsStr + str(joinedDomains[i])
                formStr = formStr + "I3"
            formStr = formStr + ",A,E19.11)'"
            domainsStr = domainsStr + ",'):"
        else:
            formStr = "'(A,E16.8)'"
            domainsStr = "scalar"
        result = result + "write(0,%s) '%s@%s', hf_output_temp\n" %(formStr, symbol.name, domainsStr)
    result = result + "write(0,*) '**********************************************'\n"
    result = result + "write(0,*) ''\n"
    if calleeRoutineNode.getAttribute('parallelRegionPosition') == 'outside':
        result += "end if\n"
        result += "#endif\n"
    return result

class FortranImplementation(object):
    architecture = "CPU"
    onDevice = False
    multipleParallelRegionsPerSubroutineAllowed = True
    optionFlags = []
    currDependantSymbols = None
    currParallelRegionTemplateNode = None
    debugPrintIteratorDeclared = False
    currRoutineNode = None
    useOpenACCForDebugPrintStatements = True

    def __init__(self, optionFlags):
        self.currDependantSymbols = None
        self.currParallelRegionTemplateNode = None
        if type(optionFlags) == list:
            self.optionFlags = optionFlags

    def filePreparation(self, filename):
        return '''#include "storage_order.F90"\n'''

    def warningOnUnrecognizedSubroutineCallInParallelRegion(self, callerName, calleeName):
        return ""

    def callPreparationForPassedSymbol(self, currRoutineNode, symbolInCaller, symbolInCallee):
        return ""

    def callPostForPassedSymbol(self, currRoutineNode, symbolInCaller, symbolInCallee):
        return ""

    def kernelCallConfig(self):
        return ""

    def kernelCallPreparation(self, parallelRegionTemplate, calleeNode=None):
        self.currParallelRegionTemplateNode = parallelRegionTemplate
        return ""

    def kernelCallPost(self, symbolsByName, calleeRoutineNode):
        if calleeRoutineNode.getAttribute('parallelRegionPosition') not in ['within', 'outside']:
            return ""
        result = ""
        if 'DEBUG_PRINT' in self.optionFlags:
            result += getRuntimeDebugPrintStatements(
                symbolsByName,
                calleeRoutineNode,
                self.currParallelRegionTemplateNode,
                useOpenACC=self.useOpenACCForDebugPrintStatements
            )
        self.currParallelRegionTemplateNode = None
        return result

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

    def getAdditionalKernelParameters(self, cgDoc, routineNode, moduleNode, parallelRegionTemplates):
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

    def declarationEndPrintStatements(self):
        return ""

    def processModuleBegin(self, moduleName):
        pass

    def processModuleEnd(self):
        pass

    def parallelRegionBegin(self, parallelRegionTemplate, outerBranchLevel=0):
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

    def parallelRegionEnd(self, parallelRegionTemplate, outerBranchLevel=0):
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
        self.currRoutineNode = currRoutineNode
        result = ""
        if 'DEBUG_PRINT' in self.optionFlags:
            result += "real(8) :: hf_output_temp\n"
            result += "#ifndef GPU\n"
            result += "integer(4), save :: hf_debug_print_iterator = 0\n"
            result += "#endif\n"
            self.debugPrintIteratorDeclared = True
            # if currRoutineNode.getAttribute('parallelRegionPosition') != 'inside':
            #     result += "#endif\n"
        return result + getIteratorDeclaration(currRoutineNode, currParallelRegionTemplates, ["CPU", ""])

    def subroutineExitPoint(self, dependantSymbols, routineIsKernelCaller, is_subroutine_end):
        result = ""
        if 'DEBUG_PRINT' in self.optionFlags and self.debugPrintIteratorDeclared:
            result += "#ifndef GPU\n"
            result += "hf_debug_print_iterator = hf_debug_print_iterator + 1\n"
            result += "#endif\n"
        if is_subroutine_end:
            self.currRoutineNode = None
            self.debugPrintIteratorDeclared = False
        return result

def getWriteTraceFunc(begin_or_end):
    def writeTrace(currRoutineNode, currModuleName, symbol):
        return "write(hf_tracing_current_path, '(A,I3.3,A)') './datatrace/%s', hf_tracing_counter, '.dat'\n" %(
            tracingFilename(currModuleName, currRoutineNode, symbol, begin_or_end)
        ) + "call writeToFile(hf_tracing_current_path, hf_tracing_temp_%s)\n" %(
            symbol.name
        )
    return writeTrace

class TraceGeneratingFortranImplementation(FortranImplementation):
    patterns = None
    currRoutineNode = None
    currModuleName = None
    currentTracedSymbols = []
    earlyReturnCounter = 0

    def __init__(self, optionFlags):
        self.patterns = H90RegExPatterns.Instance()
        self.currentTracedSymbols = []
        self.earlyReturnCounter = 0

    def additionalIncludes(self):
        return "use helper_functions\n"

    def processModuleBegin(self, moduleName):
        self.currModuleName = moduleName

    def processModuleEnd(self):
        self.currModuleName = None

    def subroutinePrefix(self, routineNode):
        self.currRoutineNode = routineNode
        return FortranImplementation.subroutinePrefix(self, routineNode)

    def declarationEnd(self, dependantSymbols, routineIsKernelCaller, currRoutineNode, currParallelRegionTemplates):
        super_result = FortranImplementation.declarationEnd(self, dependantSymbols, routineIsKernelCaller, currRoutineNode, currParallelRegionTemplates)
        result, tracedSymbols = getTracingDeclarationStatements(currRoutineNode, dependantSymbols, self.patterns)
        self.currentTracedSymbols = tracedSymbols
        sys.stderr.write("...In subroutine %s: Symbols declared for tracing: %s\n" %(
                currRoutineNode.getAttribute('name'),
                [symbol.name for symbol in tracedSymbols],
            )
        )
        return result + super_result + getTracingStatements(
            self.currRoutineNode,
            self.currModuleName,
            [symbol for symbol in self.currentTracedSymbols if symbol.intent in ['in', 'inout']],
            getWriteTraceFunc('begin'),
            increment_tracing_counter=False,
            loop_name_postfix='start'
        )

    def subroutineExitPoint(self, dependantSymbols, routineIsKernelCaller, is_subroutine_end):
        if not is_subroutine_end:
            self.earlyReturnCounter += 1
        result = getTracingStatements(
            self.currRoutineNode,
            self.currModuleName,
            [symbol for symbol in self.currentTracedSymbols if symbol.intent in ['out', 'inout', '', None]],
            getWriteTraceFunc('end'),
            increment_tracing_counter=len(self.currentTracedSymbols) > 0,
            loop_name_postfix='end' if is_subroutine_end else 'exit%i' %(self.earlyReturnCounter)
        )
        sys.stderr.write("...In subroutine %s: Symbols %s used for tracing\n" %(
                self.currRoutineNode.getAttribute('name'),
                [symbol.name for symbol in self.currentTracedSymbols]
            )
        )
        if is_subroutine_end:
            self.earlyReturnCounter = 0
            self.currentTracedSymbols = []
        return result + FortranImplementation.subroutineExitPoint(self, dependantSymbols, routineIsKernelCaller, is_subroutine_end)

class OpenMPFortranImplementation(FortranImplementation):
    def __init__(self, optionFlags):
        self.currDependantSymbols = None
        FortranImplementation.__init__(self, optionFlags)

    def declarationEnd(self, dependantSymbols, routineIsKernelCaller, currRoutineNode, currParallelRegionTemplates):
        self.currDependantSymbols = dependantSymbols
        return FortranImplementation.declarationEnd(self, dependantSymbols, routineIsKernelCaller, currRoutineNode, currParallelRegionTemplates)

    def parallelRegionBegin(self, parallelRegionTemplate, outerBranchLevel=0):
        if self.currDependantSymbols == None:
            raise Exception("parallel region without any dependant arrays")
        openMPLines = "!$OMP PARALLEL DO DEFAULT(firstprivate) %s " %(getReductionClause(parallelRegionTemplate).upper())
        openMPLines += "SHARED(%s)\n" %(', '.join([symbol.nameInScope() for symbol in self.currDependantSymbols]))
        return openMPLines + FortranImplementation.parallelRegionBegin(self, parallelRegionTemplate)

    def parallelRegionEnd(self, parallelRegionTemplate, outerBranchLevel=0):
        openMPLines = "\n!$OMP END PARALLEL DO\n"
        return FortranImplementation.parallelRegionEnd(self, parallelRegionTemplate) + openMPLines

    def subroutineExitPoint(self, dependantSymbols, routineIsKernelCaller, is_subroutine_end):
        if is_subroutine_end:
            self.currDependantSymbols = None
        return FortranImplementation.subroutineExitPoint(self, dependantSymbols, routineIsKernelCaller, is_subroutine_end)

class PGIOpenACCFortranImplementation(FortranImplementation):
    architecture = "GPU"
    onDevice = True
    currRoutineHasDataDeclarations = False
    createDeclaration = "create"
    currDependantSymbols = None

    def __init__(self, optionFlags):
        FortranImplementation.__init__(self, optionFlags)
        self.currRoutineNode = None
        self.currRoutineHasDataDeclarations = False
        self.createDeclaration = "create"
        self.currDependantSymbols = None
        self.currParallelRegionTemplates = None

    def filePreparation(self, filename):
        additionalStatements = '''
attributes(global) subroutine HF_DUMMYKERNEL_%s()
use cudafor
!This ugly hack is used because otherwise as of PGI 14.7, OpenACC kernels could not be used in code that is compiled with CUDA flags.
end subroutine
        ''' %(os.path.basename(filename).split('.')[0])
        return FortranImplementation.filePreparation(self, filename) + additionalStatements

    def additionalIncludes(self):
        return FortranImplementation.additionalIncludes(self) + "use openacc\nuse cudafor\n"

    def callPreparationForPassedSymbol(self, currRoutineNode, symbolInCaller, symbolInCallee):
        if not currRoutineNode:
            return ""
        if currRoutineNode.getAttribute("parallelRegionPosition") != 'inside':
            return ""
        if symbolInCaller.declarationType() != DeclarationType.LOCAL_ARRAY:
            return ""
        return "!$acc update device(%s) if(hf_symbols_are_device_present)\n" %(symbolInCaller.name)

    def callPostForPassedSymbol(self, currRoutineNode, symbolInCaller, symbolInCallee):
        if not currRoutineNode:
            return ""
        if currRoutineNode.getAttribute("parallelRegionPosition") != 'inside':
            return ""
        if symbolInCaller.declarationType() != DeclarationType.LOCAL_ARRAY:
            return ""
        return "!$acc update host(%s) if(hf_symbols_are_device_present)\n" %(symbolInCaller.name)

    def adjustDeclarationForDevice(self, line, patterns, dependantSymbols, routineIsKernelCaller, parallelRegionPosition):
        return line

    def declarationEnd(self, dependantSymbols, routineIsKernelCaller, currRoutineNode, currParallelRegionTemplates):
        presentDeclaration = "present" # if currRoutineNode.getAttribute("parallelRegionPosition") == 'inside' else "deviceptr"
        self.currRoutineNode = currRoutineNode
        self.currDependantSymbols = dependantSymbols
        self.currParallelRegionTemplates = currParallelRegionTemplates
        dataDirective, dataDeclarationsRequired = getDataDirectiveAndUpdateOnDeviceFlags(
            currRoutineNode,
            currParallelRegionTemplates,
            dependantSymbols,
            self.createDeclaration,
            routineIsKernelCaller,
            debugPrint=False,
            enterOrExit='enter'
        )
        self.currRoutineHasDataDeclarations = dataDeclarationsRequired
        result = ""
        if 'DEBUG_PRINT' in self.optionFlags:
            result += "real(8) :: hf_output_temp\n"
        result += getIteratorDeclaration(currRoutineNode, currParallelRegionTemplates, ["GPU"])
        result += "integer(4) :: hf_symbols_are_device_present\n"
        if dataDeclarationsRequired == True:
            result += dataDirective
        devicePresentSymbols = [symbol for symbol in dependantSymbols if symbol.isOnDevice]
        if len(devicePresentSymbols) > 0:
            for symbol in dependantSymbols:
                if len(symbol.domains) > 0 and symbol.declarationType() != DeclarationType.LOCAL_ARRAY:
                    result += "hf_symbols_are_device_present = acc_is_present(%s)\n" %(symbol.name)
                    break
            else:
                pass
                # if currRoutineNode.getAttribute("parallelRegionPosition") in ['inside', 'within']:
                #     raise Exception("kernel or kernel wrapper only has temporary data on the device - no input or output possible here.")
        result += self.declarationEndPrintStatements()
        return result

    def getIterators(self, parallelRegionTemplate):
        if not appliesTo(["GPU"], parallelRegionTemplate):
            return []
        return [domain.name for domain in getDomainsWithParallelRegionTemplate(parallelRegionTemplate)]

    def loopPreparation(self):
        return "!$acc loop seq"

    def parallelRegionBegin(self, parallelRegionTemplate, outerBranchLevel=0):
        regionStr = ""
        for symbol in self.currDependantSymbols:
            if symbol.declarationType() == DeclarationType.LOCAL_ARRAY:
                regionStr += "!$acc update device(%s) if(hf_symbols_are_device_present)\n" %(symbol.name)
        vectorSizePPNames = getVectorSizePPNames(parallelRegionTemplate)
        regionStr += "!$acc kernels if(hf_symbols_are_device_present) "
        for symbol in self.currDependantSymbols:
            if symbol.isOnDevice:
                regionStr += "present(%s) " %(symbol.name)
        regionStr += "\n"
        domains = getDomainsWithParallelRegionTemplate(parallelRegionTemplate)
        if len(domains) > 3 or len(domains) < 1:
            raise Exception("Invalid number of parallel domains in parallel region definition.")
        for pos in range(len(domains)-1,-1,-1): #use inverted order (optimization of accesses for fortran storage order)
            regionStr += "!$acc loop independent vector(%s) " %(vectorSizePPNames[pos])
            # reduction clause is broken in 15.3. better to let the compiler figure it out.
            # if pos == len(domains)-1:
            #     regionStr += getReductionClause(parallelRegionTemplate)
            regionStr += "\n"
            domain = domains[pos]
            startsAt = domain.startsAt if domain.startsAt != None else "1"
            endsAt = domain.endsAt if domain.endsAt != None else domain.size
            regionStr += 'do %s=%s,%s' %(domain.name, startsAt, endsAt)
            if pos != 0:
                regionStr += '\n '
        return regionStr

    def parallelRegionEnd(self, parallelRegionTemplate, outerBranchLevel=0):
        additionalStatements = "\n!$acc end kernels\n"
        for symbol in self.currDependantSymbols:
            if symbol.declarationType() == DeclarationType.LOCAL_ARRAY:
                additionalStatements += "!$acc update host(%s) if(hf_symbols_are_device_present)\n" %(symbol.name)
        return FortranImplementation.parallelRegionEnd(self, parallelRegionTemplate) + additionalStatements

    #MMU: we first need a branch analysis on the subroutine to do this
    # def subroutinePostfix(self, routineNode):
    #     parallelRegionPosition = routineNode.getAttribute("parallelRegionPosition")
    #     if parallelRegionPosition == "outside":
    #         return "!$acc routine seq\n"
    #     return ""

    def subroutineExitPoint(self, dependantSymbols, routineIsKernelCaller, is_subroutine_end):
        result = ""
        if self.currRoutineHasDataDeclarations:
            dataDirective, _ = getDataDirectiveAndUpdateOnDeviceFlags(
                self.currRoutineNode,
                self.currParallelRegionTemplates,
                dependantSymbols,
                self.createDeclaration,
                routineIsKernelCaller,
                'DEBUG_PRINT' in self.optionFlags,
                enterOrExit='exit'
            )
            result += dataDirective
        if is_subroutine_end:
            self.currRoutineNode = None
            self.currDependantSymbols = None
            self.currRoutineHasDataDeclarations = False
            self.currParallelRegionTemplates = None
        return result + FortranImplementation.subroutineExitPoint(self, dependantSymbols, routineIsKernelCaller, is_subroutine_end)

class DebugPGIOpenACCFortranImplementation(PGIOpenACCFortranImplementation):

    def kernelCallPreparation(self, parallelRegionTemplate, calleeNode=None):
        result = PGIOpenACCFortranImplementation.kernelCallPreparation(self, parallelRegionTemplate, calleeNode)
        if calleeNode != None:
            routineName = calleeNode.getAttribute('name')
            result += "write(0,*) 'calling kernel %s'\n" %(routineName)
        return result

    def declarationEndPrintStatements(self):
        if self.currRoutineNode.getAttribute('parallelRegionPosition') == 'outside':
            return ""
        result = PGIOpenACCFortranImplementation.declarationEndPrintStatements(self)
        routineName = self.currRoutineNode.getAttribute('name')
        result += "write(0,*) 'entering subroutine %s'\n" %(routineName)
        return result

class TraceCheckingOpenACCFortranImplementation(DebugPGIOpenACCFortranImplementation):
    patterns = None
    currRoutineNode = None
    currModuleName = None
    currentTracedSymbols = []
    earlyReturnCounter = 0

    def __init__(self, optionFlags):
        DebugPGIOpenACCFortranImplementation.__init__(self, optionFlags)
        self.patterns = H90RegExPatterns.Instance()
        self.currentTracedSymbols = []

    def additionalIncludes(self):
        return DebugPGIOpenACCFortranImplementation.additionalIncludes(self) + "use helper_functions\nuse cudafor\n"

    def processModuleBegin(self, moduleName):
        self.currModuleName = moduleName

    def processModuleEnd(self):
        self.currModuleName = None

    def subroutinePrefix(self, routineNode):
        self.currRoutineNode = routineNode
        return DebugPGIOpenACCFortranImplementation.subroutinePrefix(self, routineNode)

    def declarationEnd(self, dependantSymbols, routineIsKernelCaller, currRoutineNode, currParallelRegionTemplates):
        openACCDeclarations = DebugPGIOpenACCFortranImplementation.declarationEnd(self, dependantSymbols, routineIsKernelCaller, currRoutineNode, currParallelRegionTemplates)
        result = "integer(4) :: hf_tracing_imt, hf_tracing_ierr\n"
        result += "real(8) :: hf_tracing_error\n"
        result += "real(8) :: hf_mean_ref\n"
        result += "real(8) :: hf_mean_gpu\n"
        result += "integer(8) :: hf_num_of_elements\n"
        result += "logical :: hf_tracing_error_found\n"
        tracing_declarations, tracedSymbols = getTracingDeclarationStatements(
            currRoutineNode,
            dependantSymbols,
            self.patterns,
            useReorderingByAdditionalSymbolPrefixes={'hf_tracing_temp_':False, 'hf_tracing_comparison_':False}
        )
        result += tracing_declarations
        result += openACCDeclarations
        self.currentTracedSymbols = tracedSymbols
        result += "hf_tracing_error_found = .false.\n"
        result += "hf_tracing_error = 0.0d0\n"
        result += getTracingStatements(
            self.currRoutineNode,
            self.currModuleName,
            [symbol for symbol in self.currentTracedSymbols if symbol.intent in ['in', 'inout']],
            getCompareToTraceFunc(abortSubroutineOnError=False, loop_name_postfix='start', begin_or_end='begin'),
            increment_tracing_counter=False,
            loop_name_postfix='start'
        )
        # if len(self.currentTracedSymbols) > 0:
        #     result += "if (hf_tracing_error_found) then\n"
        #     result += "stop 2\n"
        #     result += "end if\n"

        return getIteratorDeclaration(currRoutineNode, currParallelRegionTemplates, ["GPU"]) + result

    def subroutineExitPoint(self, dependantSymbols, routineIsKernelCaller, is_subroutine_end):
        if not is_subroutine_end:
            self.earlyReturnCounter += 1
        result = getTracingStatements(
            self.currRoutineNode,
            self.currModuleName,
            [symbol for symbol in self.currentTracedSymbols if symbol.intent in ['out', 'inout', '', None]],
            getCompareToTraceFunc(
                abortSubroutineOnError=False,
                loop_name_postfix='end' if is_subroutine_end else 'exit%i' %(self.earlyReturnCounter),
                begin_or_end='end'
            ),
            increment_tracing_counter=len(self.currentTracedSymbols) > 0,
            loop_name_postfix='end' if is_subroutine_end else 'exit%i' %(self.earlyReturnCounter)
        )
        # if len(self.currentTracedSymbols) > 0:
        #     result += "if (hf_tracing_error_found) then\n"
        #     result += "stop 2\n"
        #     result += "end if\n"
        result += DebugPGIOpenACCFortranImplementation.subroutineExitPoint(self, dependantSymbols, routineIsKernelCaller, is_subroutine_end)
        if is_subroutine_end:
            self.earlyReturnCounter = 0
            self.currRoutineNode = None
            self.currentTracedSymbols = []
        return result

class CUDAFortranImplementation(FortranImplementation):
    architecture = "GPU"
    onDevice = True
    multipleParallelRegionsPerSubroutineAllowed = False
    useOpenACCForDebugPrintStatements = False

    def __init__(self, optionFlags):
        self.currRoutineNode = None
        FortranImplementation.__init__(self, optionFlags)

    def warningOnUnrecognizedSubroutineCallInParallelRegion(self, callerName, calleeName):
        return "WARNING: subroutine %s called inside %s's parallel region, but it is not defined in a h90 file.\n" \
                    %(calleeName, callerName)

    def kernelCallConfig(self):
        return "<<< cugrid, cublock >>>"

    def kernelCallPreparation(self, parallelRegionTemplate, calleeNode=None):
        result = FortranImplementation.kernelCallPreparation(self, parallelRegionTemplate, calleeNode)
        if not appliesTo(["GPU"], parallelRegionTemplate):
            return ""
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
        result += gridPreparationStr + "\n" + gridStr + ")\n" + blockStr + ")\n"
        return result

    def kernelCallPost(self, symbolsByName, calleeRoutineNode):
        result = FortranImplementation.kernelCallPost(self, symbolsByName, calleeRoutineNode)
        if calleeRoutineNode.getAttribute('parallelRegionPosition') != 'within':
            return result
        result += getCUDAErrorHandling(calleeRoutineNode)
        return result

    def getAdditionalKernelParameters(self, cgDoc, routineNode, moduleNode, parallelRegionTemplates):
        def getAdditionalImportsAndDeclarationsForParentScope(parentNode):
            additionalImports = []
            additionalDeclarations = []
            if not parallelRegionTemplates:
                return additionalImports, additionalDeclarations
            if not parentNode.getAttribute("parallelRegionPosition") == "within":
                return additionalImports, additionalDeclarations

            dependantTemplatesAndEntries = getDomainDependantTemplatesAndEntries(cgDoc, parentNode)
            for template, entry in dependantTemplatesAndEntries:
                dependantName = entry.firstChild.nodeValue
                symbol = Symbol(dependantName, template)
                symbol.loadDomainDependantEntryNodeAttributes(entry)

                #check for external module imports in kernel subroutines
                if symbol.sourceModule and symbol.sourceModule != "":
                    symbol.loadRoutineNodeAttributes(parentNode, parallelRegionTemplates)
                    additionalImports.append(symbol)
                #check for temporary arrays in kernel subroutines
                elif not symbol.intent or symbol.intent in ["", None, "local"]:
                    symbol.loadRoutineNodeAttributes(parentNode, parallelRegionTemplates)
                    if symbol.intent == "local" and len(symbol.domains) == 0:
                        continue
                    additionalDeclarations.append(symbol)
            return additionalImports, additionalDeclarations

        routineImports, routineDeclarations = getAdditionalImportsAndDeclarationsForParentScope(routineNode)
        moduleImports, moduleDeclarations = getAdditionalImportsAndDeclarationsForParentScope(moduleNode)
        return sorted(routineImports + moduleImports), sorted(routineDeclarations + moduleDeclarations)

    def extractListOfAdditionalSubroutineSymbols(self, routineNode, currSymbolsByName):
        result = []
        if not routineNode.getAttribute("parallelRegionPosition") == "within":
            return result
        for symbolName in currSymbolsByName.keys():
            symbol = currSymbolsByName[symbolName]
            if symbol.sourceModule and symbol.sourceModule != "":
                result.append(symbol)
            elif not symbol.intent or symbol.intent in ["", None] or symbol.intent == "local" and len(symbol.domains) > 0:
                #we got temporary arrays in a kernel or local module scalars -> need to be handled by framework
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

    def subroutineExitPoint(self, dependantSymbols, routineIsKernelCaller, is_subroutine_end):
        if is_subroutine_end:
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
                    deviceInitStatements += "deallocate(%s)\n" %(symbol.nameInScope())
        return deviceInitStatements + FortranImplementation.subroutineExitPoint(self, dependantSymbols, routineIsKernelCaller, is_subroutine_end)

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

        declarationDirectivesWithoutIntent, declarationDirectives,  symbolDeclarationStr = purgeFromDeclarationSettings( \
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
            elif (symbol.isPresent and alreadyOnDevice == "no") \
            or (not symbol.isPresent and alreadyOnDevice == "yes"):
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
        isOnHost = "undefined"
        for symbol in dependantSymbols:
            if not symbol.domains or len(symbol.domains) == 0:
                continue
            elif symbol.isHostSymbol and isOnHost == "undefined":
                isOnHost = "yes"
            elif not symbol.isHostSymbol and isOnHost == "undefined":
                isOnHost = "no"
            elif (symbol.isHostSymbol and symbol.isHostSymbol == "no") or (not symbol.isHostSymbol and symbol.isHostSymbol == "yes"):
                raise Exception("Declaration line contains a mix of host / non host arrays. \
Symbols vs host attributes:\n%s" %(str([(symbol.name, symbol.isHostSymbol) for symbol in dependantSymbols])) \
                )
        if copyHere == "yes" and alreadyOnDevice == "yes":
            raise Exception("Symbols with 'present' attribute cannot appear on the same specification line as symbols with 'transferHere' attribute.")
        if copyHere == "yes" and isOnHost == "yes":
            raise Exception("Symbols with 'transferHere' attribute cannot appear on the same specification line as symbols with 'host' attribute.")
        if alreadyOnDevice == "yes" and isOnHost == "yes":
            raise Exception("Symbols with 'present' attribute cannot appear on the same specification line as symbols with 'host' attribute.")

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
        and (declarationType == DeclarationType.FOREIGN_MODULE_SCALAR or declarationType == DeclarationType.LOCAL_MODULE_SCALAR):
            adjustedLine = declarationDirectives + " ,intent(in), value ::" + symbolDeclarationStr

        #local arrays in kernels
        elif parallelRegionPosition == "within" \
        and declarationType == DeclarationType.LOCAL_ARRAY:
            adjustedLine = declarationDirectives + ",intent(out), device ::" + symbolDeclarationStr

        #passed in scalars in kernels and inside kernels
        elif parallelRegionPosition in ["within", "outside"] \
        and len(dependantSymbols[0].domains) == 0 \
        and intent not in ["out", "inout", "local"]:
            #handle scalars (passed by value)
            adjustedLine = declarationDirectives + " ,value ::" + symbolDeclarationStr
            for dependantSymbol in dependantSymbols:
                dependantSymbol.isOnDevice = True

        #arrays outside of kernels
        elif len(dependantSymbols[0].domains) > 0:
            if isOnHost == "yes":
                for dependantSymbol in dependantSymbols:
                    dependantSymbol.isOnDevice = False
            elif alreadyOnDevice == "yes" or intent in [None, "", "local"]:
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

    def parallelRegionBegin(self, parallelRegionTemplate, outerBranchLevel=0):
        # if appliesTo(["GPU"], parallelRegionTemplate) and outerBranchLevel != 0:
        #     raise Exception("Cannot implement a GPU parallel region inside a branch. Please move this parallel region into its own subroutine.")
        domains = getDomainsWithParallelRegionTemplate(parallelRegionTemplate)
        regionStr = self.iteratorDefinitionBeforeParallelRegion(domains)
        regionStr += self.safetyOutsideRegion(domains)
        return regionStr

    def parallelRegionEnd(self, parallelRegionTemplate, outerBranchLevel=0):
        return ""

    def declarationEnd(self, dependantSymbols, routineIsKernelCaller, currRoutineNode, currParallelRegionTemplates):
        self.currRoutineNode = currRoutineNode
        result = ""
        if 'DEBUG_PRINT' in self.optionFlags:
            result += "real(8) :: hf_output_temp\n"
        result += getIteratorDeclaration(currRoutineNode, currParallelRegionTemplates, ["GPU"])

        if routineIsKernelCaller:
            result += "type(dim3) :: cugrid, cublock\n"
            result += "integer(4) :: cugridSizeX, cugridSizeY, cugridSizeZ, cuerror, cuErrorMemcopy\n"

        result += self.declarationEndPrintStatements()

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

    def kernelCallPreparation(self, parallelRegionTemplate, calleeNode=None):
        result = CUDAFortranImplementation.kernelCallPreparation(self, parallelRegionTemplate, calleeNode)
        if calleeNode != None:
            iterators = self.getIterators(parallelRegionTemplate)
            gridSizeVarNames = ["cugridSizeX", "cugridSizeY", "cugridSizeZ"]
            routineName = calleeNode.getAttribute('name')
            result += "write(0,*) 'calling kernel %s with grid size', " %(routineName)
            for i in range(len(iterators)):
                if i != 0:
                    result += ", "
                result += "%s" %(gridSizeVarNames[i])
            result += "\n"
        return result

    def kernelCallPost(self, symbolsByName, calleeRoutineNode):
        result = CUDAFortranImplementation.kernelCallPost(self, symbolsByName, calleeRoutineNode)
        if calleeRoutineNode.getAttribute('parallelRegionPosition') != 'within':
            return result
        result += "if(cuerror .NE. cudaSuccess) then\n"\
                "\tstop 1\n" \
            "end if\n"
        return result

    def declarationEndPrintStatements(self):
        if self.currRoutineNode.getAttribute('parallelRegionPosition') != 'inside':
            return ""
        result = CUDAFortranImplementation.declarationEndPrintStatements(self)
        routineName = self.currRoutineNode.getAttribute('name')
        result += "write(0,*) 'entering subroutine %s'\n" %(routineName)
        return result

class DebugEmulatedCUDAFortranImplementation(DebugCUDAFortranImplementation):

    def __init__(self, optionFlags):
        self.currDependantSymbols = None
        DebugCUDAFortranImplementation.__init__(self, optionFlags)

    def declarationEnd(self, dependantSymbols, routineIsKernelCaller, currRoutineNode, currParallelRegionTemplates):
        self.currDependantSymbols = dependantSymbols
        return DebugCUDAFortranImplementation.declarationEnd(self, dependantSymbols, routineIsKernelCaller, currRoutineNode, currParallelRegionTemplates)

    def parallelRegionBegin(self, parallelRegionTemplate, outerBranchLevel=0):
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

    def subroutineExitPoint(self, dependantSymbols, routineIsKernelCaller, is_subroutine_end):
        if is_subroutine_end:
            self.currDependantSymbols = None
        return DebugCUDAFortranImplementation.subroutineExitPoint(self, dependantSymbols, routineIsKernelCaller, is_subroutine_end)




