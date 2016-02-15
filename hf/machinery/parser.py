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

import os, sys, fileinput, re, traceback, logging
from tools.metadata import *
from models.symbol import *
from tools.commons import UsageError, BracketAnalyzer
from tools.analysis import SymbolDependencyAnalyzer, getAnalysisForSymbol, getArguments
from tools.patterns import RegExPatterns
from machinery.commons import FortranRoutineArgumentParser, FortranCodeSanitizer

currFile = None
currLineNo = None

class CallGraphParser(object):
    '''An imperative python parser for h90 sourcefiles, based on a finite state machine and regex'''
    '''This class is intended as an abstract class to be inherited and doesn't do anything useful by itself'''
    '''A minimal implementation of this class implements one or more process*Match routines'''
    state = 'none'
    currSubprocName = None
    currModuleName = None
    currArgumentParser = None
    currCalleeName = None
    currArguments = None
    patterns = None
    branchAnalyzer = None
    currTemplateName = None
    stateSwitch = None
    currSymbolsByName = None

    def __init__(self):
        self.patterns = RegExPatterns.Instance()
        self.state = "none"
        self.currCalleeName = None
        self.currArguments = None
        self.currModuleName = None
        self.currSymbolsByName = {}
        self.branchAnalyzer = BracketAnalyzer(
            r'^\s*if\s*\(|^\s*select\s+case',
            r'^\s*end\s+if|^\s*end\s+select',
            pass_in_regex_pattern=True
        )
        self.stateSwitch = {
           'none': self.processNoneState,
           'inside_module': self.processInsideModuleState,
           'inside_moduleDomainDependantRegion': self.processInsideModuleDomainDependantRegionState,
           'inside_module_body': self.processInsideModuleBodyState,
           'inside_declarations': self.processInsideDeclarationsState,
           'inside_parallelRegion': self.processInsideParallelRegionState,
           'inside_domainDependantRegion': self.processInsideDomainDependantRegionState,
           'inside_subroutine_body': self.processInsideSubroutineBodyState,
           'inside_branch': self.processInsideBranch,
           'inside_ignore': self.processInsideIgnore
         }
        super(CallGraphParser, self).__init__()

    def processCallMatch(self, subProcCallMatch):
        if (not subProcCallMatch.group(1) or subProcCallMatch.group(1) == ''):
            raise UsageError("subprocedure call without matching subprocedure name")
        self.currArgumentParser = FortranRoutineArgumentParser()
        self.currArgumentParser.processString(subProcCallMatch.group(0), self.patterns)
        self.currArguments = self.currArgumentParser.arguments
        self.currCalleeName = subProcCallMatch.group(1)
        return

    def processCallPost(self):
        self.currArguments = None
        self.currArgumentParser = None
        self.currCalleeName = None

    def processBranchMatch(self, branchMatch):
        return

    def processProcBeginMatch(self, subProcBeginMatch):
        logging.debug('entering %s' %(subProcBeginMatch.group(1)), extra={"hfLineNo":currLineNo, "hfFile":currFile})
        self.currSubprocName = subProcBeginMatch.group(1)
        self.currArgumentParser = FortranRoutineArgumentParser()
        self.currArgumentParser.processString(subProcBeginMatch.group(0), self.patterns)
        self.currArguments = self.currArgumentParser.arguments
        return

    def processProcEndMatch(self, subProcEndMatch):
        logging.debug('exiting subprocedure', extra={"hfLineNo":currLineNo, "hfFile":currFile})
        self.currSubprocName = None
        self.currArgumentParser = None
        self.currArguments = None
        return

    def processParallelRegionMatch(self, parallelRegionMatch):
        return

    def processParallelRegionEndMatch(self, parallelRegionEndMatch):
        return

    def processDomainDependantMatch(self, domainDependantMatch):
        return

    def processDomainDependantEndMatch(self, domainDependantEndMatch):
        return

    def processModuleBeginMatch(self, moduleBeginMatch):
        return

    def processModuleEndMatch(self, moduleEndMatch):
        return

    def processTemplateMatch(self, templateMatch):
        settingPattern = re.compile(r'[\s,]*(\w*)\s*(\(.*)')
        settingMatch = settingPattern.match(templateMatch.group(1))
        if not settingMatch:
            self.currTemplateName = None
            return
        settingName = settingMatch.group(1).strip()
        if settingName != 'name':
            self.currTemplateName = None
            return
        textAfterSettingName = settingMatch.group(2)
        settingBracketAnalyzer = BracketAnalyzer()
        settingText, remainder = settingBracketAnalyzer.getTextWithinBracketsAndRemainder(textAfterSettingName)
        self.currTemplateName = settingText

    def processTemplateEndMatch(self, templateEndMatch):
        self.currTemplateName = None

    def processContainsMatch(self, containsMatch):
        return

    def processNoMatch(self, line):
        return

    def processNoneState(self, line):
        moduleBeginMatch = self.patterns.moduleBeginPattern.match(str(line))
        subProcBeginMatch = self.patterns.subprocBeginPattern.match(str(line))
        templateMatch = self.patterns.templatePattern.match(str(line))
        templateEndMatch = self.patterns.templateEndPattern.match(str(line))
        branchMatch = self.patterns.branchPattern.match(str(line))
        if branchMatch:
            self.processBranchMatch(branchMatch)
        elif templateMatch:
            self.processTemplateMatch(templateMatch)
        elif templateEndMatch:
            self.processTemplateEndMatch(templateEndMatch)
        elif moduleBeginMatch:
            self.currModuleName = moduleBeginMatch.group(1)
            self.state = 'inside_module'
            self.processModuleBeginMatch(moduleBeginMatch)
        elif subProcBeginMatch:
            raise UsageError("please put this Hybrid Fortran subroutine into a module")
        else:
            self.processNoMatch(line)

    def processInsideBranch(self, line):
        return

    def processInsideIgnore(self, line):
        return

    def processInsideModuleState(self, line):
        moduleEndMatch = self.patterns.moduleEndPattern.match(str(line))
        domainDependantMatch = self.patterns.domainDependantPattern.match(str(line))
        templateMatch = self.patterns.templatePattern.match(str(line))
        templateEndMatch = self.patterns.templateEndPattern.match(str(line))
        branchMatch = self.patterns.branchPattern.match(str(line))
        containsMatch = self.patterns.containsPattern.match(str(line))

        if branchMatch:
            self.processBranchMatch(branchMatch)
        elif templateMatch:
            self.processTemplateMatch(templateMatch)
        elif templateEndMatch:
            self.processTemplateEndMatch(templateEndMatch)
        elif domainDependantMatch:
            if self.state == "inside_branch":
                self.stateBeforeBranch = 'inside_moduleDomainDependantRegion'
            else:
                self.state = 'inside_moduleDomainDependantRegion'
            self.processDomainDependantMatch(domainDependantMatch)
        elif moduleEndMatch:
            self.processModuleEndMatch(moduleEndMatch)
            self.currModuleName = None
            if self.state == "inside_branch":
                self.stateBeforeBranch = 'none'
            else:
                self.state = 'none'
        elif containsMatch:
            self.processContainsMatch(containsMatch)
            if self.state == "inside_branch":
                self.stateBeforeBranch = 'inside_module_body'
            else:
                self.state = 'inside_module_body'
        else:
            self.processNoMatch(line)


    def processInsideModuleBodyState(self, line):
        subProcBeginMatch = self.patterns.subprocBeginPattern.match(str(line))
        moduleEndMatch = self.patterns.moduleEndPattern.match(str(line))
        templateMatch = self.patterns.templatePattern.match(str(line))
        templateEndMatch = self.patterns.templateEndPattern.match(str(line))
        branchMatch = self.patterns.branchPattern.match(str(line))

        if branchMatch:
            self.processBranchMatch(branchMatch)
        elif templateMatch:
            self.processTemplateMatch(templateMatch)
        elif templateEndMatch:
            self.processTemplateEndMatch(templateEndMatch)
        elif moduleEndMatch:
            self.processModuleEndMatch(moduleEndMatch)
            self.currModuleName = None
            if self.state == "inside_branch":
                self.stateBeforeBranch = 'none'
            else:
                self.state = 'none'
        elif subProcBeginMatch:
            if (not subProcBeginMatch.group(1) or subProcBeginMatch.group(1) == ''):
                raise UsageError("subprocedure begin without matching subprocedure name")
            self.processProcBeginMatch(subProcBeginMatch)
            if self.state == "inside_branch":
                self.stateBeforeBranch = 'inside_declarations'
            else:
                self.state = 'inside_declarations'
            self.processSubprocStartPost()
        elif (self.patterns.subprocEndPattern.match(str(line))):
            raise UsageError("end subprocedure without matching begin subprocedure")
        else:
            self.processNoMatch(line)

    def processSubprocStartPost(self):
        return

    def processInsideDeclarationsState(self, line):
        subProcCallMatch = self.patterns.subprocCallPattern.match(str(line))
        parallelRegionMatch = self.patterns.parallelRegionPattern.match(str(line))
        domainDependantMatch = self.patterns.domainDependantPattern.match(str(line))
        subProcEndMatch = self.patterns.subprocEndPattern.match(str(line))
        templateMatch = self.patterns.templatePattern.match(str(line))
        templateEndMatch = self.patterns.templateEndPattern.match(str(line))
        branchMatch = self.patterns.branchPattern.match(str(line))

        if branchMatch:
            self.processBranchMatch(branchMatch)
        elif (domainDependantMatch):
            if self.state == "inside_branch":
                self.stateBeforeBranch = 'inside_domainDependantRegion'
            else:
                self.state = 'inside_domainDependantRegion'
            self.processDomainDependantMatch(domainDependantMatch)
        elif subProcCallMatch:
            self.processCallMatch(subProcCallMatch)
            if (self.state == "inside_branch" and self.stateBeforeBranch != 'inside_subroutine_call') or (self.state != "inside_branch" and self.state != 'inside_subroutine_call'):
                self.processCallPost()
        elif subProcEndMatch:
            self.processProcEndMatch(subProcEndMatch)
            if self.state == "inside_branch":
                self.stateBeforeBranch = 'inside_module_body'
            else:
                self.state = 'inside_module_body'
        elif parallelRegionMatch:
            raise UsageError("parallel region without parallel dependants")
        elif (self.patterns.subprocBeginPattern.match(str(line))):
            raise UsageError("subprocedure within subprocedure not allowed")
        elif templateMatch:
            raise UsageError("template directives are only allowed outside of subroutines")
        elif templateEndMatch:
            raise UsageError("template directives are only allowed outside of subroutines")
        if self.state != "inside_declarations" and not (self.state == "inside_branch" and self.stateBeforeBranch == "inside_declarations"):
            self.currArgumentParser = None
            self.currArguments = None

    def processInsideSubroutineBodyState(self, line):
        #note: Branches (@if statements) are ignored here, we want to keep analyzing their statements for callgraphs.
        domainDependantMatch = self.patterns.domainDependantPattern.match(str(line))
        subProcCallMatch = self.patterns.subprocCallPattern.match(str(line))
        parallelRegionMatch = self.patterns.parallelRegionPattern.match(str(line))
        domainDependantMatch = self.patterns.domainDependantPattern.match(str(line))
        subProcEndMatch = self.patterns.subprocEndPattern.match(str(line))
        templateMatch = self.patterns.templatePattern.match(str(line))
        templateEndMatch = self.patterns.templateEndPattern.match(str(line))
        branchMatch = self.patterns.branchPattern.match(str(line))

        if branchMatch:
            self.processBranchMatch(branchMatch)
        elif domainDependantMatch:
            if self.state == "inside_branch":
                self.stateBeforeBranch = 'inside_domainDependantRegion'
            else:
                self.state = 'inside_domainDependantRegion'
            self.processDomainDependantMatch(domainDependantMatch)
        elif subProcCallMatch:
            self.processCallMatch(subProcCallMatch)
            if (self.state == "inside_branch" and self.stateBeforeBranch != 'inside_subroutine_call') or (self.state != "inside_branch" and self.state != 'inside_subroutine_call'):
                self.processCallPost()
        elif subProcEndMatch:
            self.processProcEndMatch(subProcEndMatch)
            if self.state == "inside_branch":
                self.stateBeforeBranch = 'inside_module_body'
            else:
                self.state = 'inside_module_body'
        elif parallelRegionMatch:
            self.processParallelRegionMatch(parallelRegionMatch)
            self.state = 'inside_parallelRegion'
        elif self.patterns.subprocBeginPattern.match(str(line)):
            raise UsageError("subprocedure within subprocedure not allowed")
        elif templateMatch:
            raise UsageError("template directives are only allowed outside of subroutines")
        elif templateEndMatch:
            raise UsageError("template directives are only allowed outside of subroutines")

    def processInsideParallelRegionState(self, line):
        subProcCallMatch = self.patterns.subprocCallPattern.match(str(line))
        parallelRegionEndMatch = self.patterns.parallelRegionEndPattern.match(str(line))
        templateMatch = self.patterns.templatePattern.match(str(line))
        templateEndMatch = self.patterns.templateEndPattern.match(str(line))
        branchMatch = self.patterns.branchPattern.match(str(line))

        newState = None
        if branchMatch:
            self.processBranchMatch(branchMatch)
        elif subProcCallMatch:
            self.processCallMatch(subProcCallMatch)
            if (self.state == "inside_branch" and self.stateBeforeBranch != 'inside_subroutine_call') or (self.state != "inside_branch" and self.state != 'inside_subroutine_call'):
                self.processCallPost()
        elif (parallelRegionEndMatch):
            self.processParallelRegionEndMatch(parallelRegionEndMatch)
            newState = "inside_subroutine_body"
        # elif (self.patterns.earlyReturnPattern.match(str(line))):
        #     raise UsageError("early return in the same subroutine within parallelRegion not allowed")
        elif (self.patterns.parallelRegionPattern.match(str(line))):
            raise UsageError("parallelRegion within parallelRegion not allowed")
        elif (self.patterns.subprocEndPattern.match(str(line))):
            raise UsageError("subprocedure end before @end parallelRegion")
        elif (self.patterns.subprocBeginPattern.match(str(line))):
            raise UsageError("subprocedure within subprocedure not allowed")
        elif templateMatch:
            raise UsageError("template directives are only allowed outside of subroutines")
        elif templateEndMatch:
            raise UsageError("template directives are only allowed outside of subroutines")
        else:
            self.processNoMatch(line)
        if newState == None:
            return
        if self.state == "inside_branch":
            self.stateBeforeBranch = newState
        else:
            self.state = newState

    def processInsideModuleDomainDependantRegionState(self, line):
        domainDependantEndMatch = self.patterns.domainDependantEndPattern.match(str(line))
        templateMatch = self.patterns.templatePattern.match(str(line))
        templateEndMatch = self.patterns.templateEndPattern.match(str(line))
        branchMatch = self.patterns.branchPattern.match(str(line))

        newState = None
        if branchMatch:
            self.processBranchMatch(branchMatch)
        elif domainDependantEndMatch:
            self.processDomainDependantEndMatch(domainDependantEndMatch)
            newState = "inside_module"
        elif (self.patterns.earlyReturnPattern.match(str(line))):
            raise UsageError("early return not allowed here")
        elif self.patterns.subprocCallPattern.match(str(line)):
            raise UsageError("subprocedure call within domainDependants not allowed")
        elif (self.patterns.parallelRegionEndPattern.match(str(line)) or self.patterns.parallelRegionPattern.match(str(line))):
            raise UsageError("parallelRegion within domainDependants not allowed")
        elif (self.patterns.subprocEndPattern.match(str(line))):
            raise UsageError("subprocedure end before @end domainDependant")
        elif (self.patterns.subprocBeginPattern.match(str(line))):
            raise UsageError("subprocedure within subprocedure not allowed")
        elif templateMatch:
            raise UsageError("template directives not allowed here")
        elif templateEndMatch:
            raise UsageError("template directives not allowed here")
        if newState == None:
            return
        if self.state == "inside_branch":
            self.stateBeforeBranch = newState
        else:
            self.state = newState

    def processInsideDomainDependantRegionState(self, line):
        domainDependantEndMatch = self.patterns.domainDependantEndPattern.match(str(line))
        templateMatch = self.patterns.templatePattern.match(str(line))
        templateEndMatch = self.patterns.templateEndPattern.match(str(line))
        branchMatch = self.patterns.branchPattern.match(str(line))

        newState = None
        if branchMatch:
            self.processBranchMatch(branchMatch)
        elif domainDependantEndMatch:
            self.processDomainDependantEndMatch(domainDependantEndMatch)
            newState = "inside_subroutine_body"
        elif (self.patterns.earlyReturnPattern.match(str(line))):
            raise UsageError("early return not allowed here")
        elif self.patterns.subprocCallPattern.match(str(line)):
            raise UsageError("subprocedure call within domainDependants not allowed")
        elif (self.patterns.parallelRegionEndPattern.match(str(line)) or self.patterns.parallelRegionPattern.match(str(line))):
            raise UsageError("parallelRegion within domainDependants not allowed")
        elif (self.patterns.subprocEndPattern.match(str(line))):
            raise UsageError("subprocedure end before @end domainDependant")
        elif (self.patterns.subprocBeginPattern.match(str(line))):
            raise UsageError("subprocedure within subprocedure not allowed")
        elif templateMatch:
            raise UsageError("template directives not allowed here")
        elif templateEndMatch:
            raise UsageError("template directives not allowed here")
        if newState == None:
            return
        if self.state == "inside_branch":
            self.stateBeforeBranch = newState
        else:
            self.state = newState

    def processUndefinedState(self, line):
        raise Exception("unexpected undefined parser state: %s" %(self.state))

    def processLine(self, line):
        global currLineNo
        currLineNo = self.lineNo

        #here we only load the current line into the branch analyzer for further use, we don't need the result of this method
        self.branchAnalyzer.currLevelAfterString(line)

        #analyse this line. handle the line according to current parser state.
        self.stateSwitch.get(self.state, self.processUndefinedState)(line)

    def processFile(self, fileName):
        self.lineNo = 1
        self.fileName = fileName
        global currFile
        currFile = os.path.basename(fileName)
        for line in fileinput.input([fileName]):
            try:
                self.processLine(line)
            except UsageError as e:
                logging.error('Error: %s' %(str(e)), extra={"hfLineNo":currLineNo, "hfFile":currFile})
                sys.exit(1)
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                logging.critical(
                    'Error when parsing file %s on line %i: %s; Print of line:%s\n' %(
                        str(fileName), self.lineNo, str(e), str(line).strip()
                    ),
                    extra={"hfLineNo":currLineNo, "hfFile":currFile}
                )
                logging.info(traceback.format_exc(), extra={"hfLineNo":currLineNo, "hfFile":currFile})
                sys.exit(1)
            self.lineNo += 1

        if (self.state != 'none'):
            logging.info(
                'Error when parsing file %s: File ended unexpectedly. Parser state: %s; Current Callee: %s; Current Subprocedure name: %s; Current Linenumber: %i; Current ArgumentParser: %s\n' %(
                    str(fileName), self.state, self.currCalleeName, self.currSubprocName, self.lineNo, str(self.currArgumentParser)
                ),
                extra={"hfLineNo":currLineNo, "hfFile":currFile}
            )
            sys.exit(1)
        del self.lineNo
        del self.fileName

class H90XMLCallGraphGenerator(CallGraphParser):
    doc = None
    routines = None
    modules = None
    templates = None
    calls = None
    currCallNode = None
    currSubprocNode = None
    currModuleNode = None
    currDomainDependantRelationNode = None
    currParallelRegionTemplateNode = None
    currParallelRegionRelationNode = None

    def __init__(self, doc):
        self.doc = doc
        self.routines = createOrGetFirstNodeWithName('routines', doc)
        self.calls = createOrGetFirstNodeWithName('calls', doc)
        self.modules = createOrGetFirstNodeWithName('modules', doc)
        self.templates = createOrGetFirstNodeWithName('implementationTemplates', doc)
        super(H90XMLCallGraphGenerator, self).__init__()

    def processCallMatch(self, subProcCallMatch):
        subProcName = subProcCallMatch.group(1)
        call = self.doc.createElement('call')
        call.setAttribute('caller', self.currSubprocName)
        call.setAttribute('callee', subProcName)
        if self.state == "inside_parallelRegion" or (self.state == "inside_branch" and self.stateBeforeBranch == "inside_parallelRegion"):
            call.setAttribute('parallelRegionPosition', 'surround')
        if (not firstDuplicateChild(self.calls, call)):
            self.calls.appendChild(call)
        self.currCallNode = call
        super(H90XMLCallGraphGenerator, self).processCallMatch(subProcCallMatch)

    def processArguments(self, nodeToAppendTo):
        arguments = self.doc.createElement('arguments')
        for symbolName in self.currArguments:
            argument = self.doc.createElement('argument')
            argument.setAttribute('symbolName', symbolName)
            arguments.appendChild(argument)
        nodeToAppendTo.appendChild(arguments)

    def processCallPost(self):
        self.processArguments(self.currCallNode)
        self.currCallNode = None
        super(H90XMLCallGraphGenerator, self).processCallPost()

    def processTemplateMatch(self, templateMatch):
        super(H90XMLCallGraphGenerator, self).processTemplateMatch(templateMatch)
        template = self.doc.createElement('implementationTemplate')
        template.setAttribute('name', self.currTemplateName)
        self.templates.appendChild(template)
        return

    def processModuleBeginMatch(self, moduleBeginMatch):
        super(H90XMLCallGraphGenerator, self).processModuleBeginMatch(moduleBeginMatch)
        module = self.doc.createElement('module')
        module.setAttribute('name', self.currModuleName)
        self.modules.appendChild(module)
        self.currModuleNode = module

    def processProcBeginMatch(self, subProcBeginMatch):
        super(H90XMLCallGraphGenerator, self).processProcBeginMatch(subProcBeginMatch)
        routine = self.doc.createElement('routine')
        routine.setAttribute('name', self.currSubprocName)
        routine.setAttribute('source', os.path.basename(self.fileName).split('.')[0])
        routine.setAttribute('module', self.currModuleName)
        routine.setAttribute('implementationTemplate', self.currTemplateName)
        self.routines.appendChild(routine)
        self.currSubprocNode = routine

    def processSubprocStartPost(self):
        self.processArguments(self.currSubprocNode)
        super(H90XMLCallGraphGenerator, self).processSubprocStartPost()

    def processParallelRegionMatch(self, parallelRegionMatch):
        super(H90XMLCallGraphGenerator, self).processParallelRegionMatch(parallelRegionMatch)
        self.currParallelRegionRelationNode, self.currParallelRegionTemplateNode = updateAndGetParallelRegionInfo(
            doc=self.doc,
            subroutineNode=self.currSubprocNode,
            parallelRegionSpecification=parallelRegionMatch.group(1),
            startLine=self.lineNo
        )

    def processParallelRegionEndMatch(self, parallelRegionEndMatch):
        super(H90XMLCallGraphGenerator, self).processParallelRegionEndMatch(parallelRegionEndMatch)
        self.currParallelRegionRelationNode.setAttribute("endLine", str(self.lineNo))
        self.currParallelRegionTemplateNode = None
        self.currParallelRegionRelationNode = None

    def processDomainDependantMatch(self, domainDependantMatch):
        super(H90XMLCallGraphGenerator, self).processDomainDependantMatch(domainDependantMatch)
        self.currDomainDependantRelationNode, _ = setTemplateInfos(
            self.doc,
            self.currModuleNode if self.state == 'inside_moduleDomainDependantRegion' or (self.state == "inside_branch" and self.stateBeforeBranch == "inside_moduleDomainDependantRegion") else self.currSubprocNode,
            domainDependantMatch.group(1),
            "domainDependantTemplates",
            "domainDependantTemplate",
            "domainDependants"
        )

    def processDomainDependantEndMatch(self, domainDependantEndMatch):
        super(H90XMLCallGraphGenerator, self).processDomainDependantEndMatch(domainDependantEndMatch)
        self.currDomainDependantRelationNode = None

    def processProcEndMatch(self, subProcEndMatch):
        super(H90XMLCallGraphGenerator, self).processProcEndMatch(subProcEndMatch)
        self.currSubprocNode = None

    def processModuleEndMatch(self, moduleEndMatch):
        super(H90XMLCallGraphGenerator, self).processModuleEndMatch(moduleEndMatch)
        self.currModuleNode = None

    def processInsideModuleDomainDependantRegionState(self, line):
        super(H90XMLCallGraphGenerator, self).processInsideModuleDomainDependantRegionState(line)
        if (self.state != 'inside_branch' and self.state != 'inside_moduleDomainDependantRegion') or (self.state == "inside_branch" and self.stateBeforeBranch != "inside_moduleDomainDependantRegion"):
            return
        addAndGetEntries(self.doc, self.currDomainDependantRelationNode, line)

    def processInsideDomainDependantRegionState(self, line):
        super(H90XMLCallGraphGenerator, self).processInsideDomainDependantRegionState(line)
        if (self.state != 'inside_branch' and self.state != 'inside_domainDependantRegion') or (self.state == "inside_branch" and self.stateBeforeBranch != "inside_domainDependantRegion"):
            return
        addAndGetEntries(self.doc, self.currDomainDependantRelationNode, line)

def getSymbolsByName(cgDoc, parentNode, parallelRegionTemplates=[], currentModuleName=None, currentSymbolsByName={}, symbolAnalysisByRoutineNameAndSymbolName={}, isModuleSymbols=False):
    patterns = RegExPatterns.Instance()
    templatesAndEntries = getDomainDependantTemplatesAndEntries(cgDoc, parentNode)
    symbolsByName = {}
    parentName = parentNode.getAttribute('name')
    if parentName in [None, '']:
        raise Exception("parent node without identifier")
    for template, entry in templatesAndEntries:
        dependantName = entry.firstChild.nodeValue
        symbol = Symbol(
            dependantName,
            template,
            symbolEntry=entry,
            scopeNode=parentNode,
            analysis=getAnalysisForSymbol(symbolAnalysisByRoutineNameAndSymbolName, parentName, dependantName),
            parallelRegionTemplates=parallelRegionTemplates
        )
        existingSymbol = symbolsByName.get(uniqueIdentifier(dependantName, parentName))
        if existingSymbol == None:
            existingSymbol = currentSymbolsByName.get(uniqueIdentifier(dependantName, parentName))
        if existingSymbol == None and currentModuleName not in [None, ""]:
            existingSymbol = currentSymbolsByName.get(uniqueIdentifier(dependantName, currentModuleName))
            if existingSymbol != None:
                symbol.resetScope(currentModuleName) #if this symbol is found in the local module, assume we are using that module symbol here. Problem: Does not cover when local scope overrides module scope.
        if existingSymbol != None:
            symbol.merge(existingSymbol)
        symbolsByName[symbol.uniqueIdentifier] = symbol
    return symbolsByName

def getModuleNodesByName(cgDoc):
    moduleNodesByName = {}
    modules = cgDoc.getElementsByTagName('module')
    for module in modules:
        moduleName = module.getAttribute('name')
        if not moduleName or moduleName == '':
            raise Exception("Module without name.")
        moduleNodesByName[moduleName] = module
    return moduleNodesByName

def getParallelRegionData(cgDoc):
    parallelRegionTemplateRelationsByProcName = {}
    parallelRegionTemplatesByProcName = {}
    routineNodesByProcName = {}
    routineNodesByModule = {}
    regionsByID = regionTemplatesByID(cgDoc, 'parallelRegionTemplate')
    routines = cgDoc.getElementsByTagName('routine')
    for routine in routines:
        procName = routine.getAttribute('name')
        if procName in [None, '']:
            raise Exception("Procedure without name.")
        routineNodesByProcName[procName] = routine
        moduleName = routine.getAttribute('module')
        if moduleName not in [None, '']:
            routinesForModule = routineNodesByModule.get(moduleName, [])
            routinesForModule.append(routine)
            routineNodesByModule[moduleName] = routinesForModule
        regionTemplates = []
        parallelRegionsParents = routine.getElementsByTagName('activeParallelRegions')
        if parallelRegionsParents and len(parallelRegionsParents) > 0:
            templateRelations = parallelRegionsParents[0].getElementsByTagName('templateRelation')
            for templateRelation in templateRelations:
                idStr = templateRelation.getAttribute('id')
                if not idStr or idStr == '':
                    raise Exception("Template relation without id attribute.")
                regionTemplate = regionsByID.get(idStr, None)
                if not regionTemplate:
                    raise Exception("Template relation id %s could not be matched in procedure '%s'" %(idStr, procName))
                regionTemplates.append(regionTemplate)
            if len(templateRelations) > 0:
                parallelRegionTemplateRelationsByProcName[procName] = templateRelations
            if len(regionTemplates) > 0:
                parallelRegionTemplatesByProcName[procName] = regionTemplates
    return parallelRegionTemplateRelationsByProcName, parallelRegionTemplatesByProcName, routineNodesByProcName, routineNodesByModule

class H90CallGraphAndSymbolDeclarationsParser(CallGraphParser):
    cgDoc = None
    symbolsOnCurrentLine = None
    importsOnCurrentLine = None
    routineNodesByProcName = None
    moduleNodesByName = None
    routineNodesByModule = None
    parallelRegionTemplatesByProcName = None
    parallelRegionTemplateRelationsByProcName = None
    #$$$ remove this in case we never enable routine domain dependant specifications for module symbols (likely)
    # tentativeModuleSymbolsByName = None

    def __init__(self, cgDoc, moduleNodesByName=None, parallelRegionData=None, implementationsByTemplateName=None):
        self.cgDoc = cgDoc
        self.symbolsOnCurrentLine = []
        self.importsOnCurrentLine = []
        #$$$ remove this in case we never enable routine domain dependant specifications for module symbols (likely)
        # self.tentativeModuleSymbolsByName = None
        if moduleNodesByName != None:
            self.moduleNodesByName = moduleNodesByName
        else:
            self.moduleNodesByName = getModuleNodesByName(cgDoc)

        if parallelRegionData == None:
            parallelRegionData = getParallelRegionData(cgDoc)
        self.implementationsByTemplateName = implementationsByTemplateName
        self.parallelRegionTemplatesByProcName = parallelRegionData[1]
        self.parallelRegionTemplateRelationsByProcName = parallelRegionData[0]
        self.routineNodesByProcName = parallelRegionData[2]
        self.routineNodesByModule = parallelRegionData[3]
        super(H90CallGraphAndSymbolDeclarationsParser, self).__init__()

    @property
    def implementation(self):
        return self.implementationForTemplateName(self.currTemplateName)

    def implementationForTemplateName(self, templateName):
        implementation = self.implementationsByTemplateName.get(templateName)
        if implementation == None:
            implementation = self.implementationsByTemplateName.get('default')
        if implementation == None:
            raise Exception("no default implementation defined")
        return implementation

    def loadSymbolsFromTemplate(self, parentNode, parallelRegionTemplates, isModuleSymbols=False):
        self.currSymbolsByName.update(getSymbolsByName(
            self.cgDoc,
            parentNode,
            parallelRegionTemplates=parallelRegionTemplates,
            currentModuleName=self.currModuleName,
            currentSymbolsByName=self.currSymbolsByName,
            isModuleSymbols=isModuleSymbols,
            symbolAnalysisByRoutineNameAndSymbolName=self.symbolAnalysisByRoutineNameAndSymbolName \
                if hasattr(self, 'symbolAnalysisByRoutineNameAndSymbolName') \
                else {}
        ))
        logging.debug(
            "Symbols loaded from template. Symbols currently active in scope: %s. Module Symbol Property: %s" %(
                str(self.currSymbolsByName.values()),
                str([self.currSymbolsByName[symbolName].isModuleSymbol for symbolName in self.currSymbolsByName.keys()])
            ),
            extra={"hfLineNo":currLineNo, "hfFile":currFile}
        )

    def analyseSymbolInformationOnCurrentLine(self, line, analyseImports=True, isModuleSpecification=False):
        # def loadAsAdditionalModuleSymbol(symbol): #$$$ remove this in case we never enable routine domain dependant specifications for module symbols (likely)
        #     symbol.isModuleSymbol = True
        #     symbol.loadModuleNodeAttributes(self.moduleNodesByName[self.currModuleName])
        #     self.symbolsOnCurrentLine.append(symbol)
        #     self.currSymbolsByName[symbol.name] = symbol

        selectiveImportMatch = self.patterns.selectiveImportPattern.match(str(line))
        if selectiveImportMatch:
            self.processImplicitForeignModuleSymbolMatch(selectiveImportMatch)

        symbolNames = self.currSymbolsByName.keys()
        for symbolName in symbolNames:
            symbol = self.currSymbolsByName[symbolName]
            declMatch = symbol.getDeclarationMatch(str(line))
            importMatch = None
            if analyseImports:
                importMatch = symbol.symbolImportPattern.match(str(line))
            if declMatch:
                self.symbolsOnCurrentLine.append(symbol)
                self.processSymbolDeclMatch(declMatch, symbol)
            elif importMatch:
                self.importsOnCurrentLine.append(symbol)
                self.processSymbolImportMatch(importMatch, symbol)

        #$$$ remove this in case we never enable routine domain dependant specifications for module symbols (likely)
        # if isModuleSpecification:
        #     symbolNames = self.tentativeModuleSymbolsByName.keys() if self.tentativeModuleSymbolsByName else []
        #     for symbolName in symbolNames:
        #         symbol = self.tentativeModuleSymbolsByName[symbolName]
        #         declMatch = symbol.getDeclarationMatch(str(line))
        #         importMatch = None
        #         if analyseImports:
        #             importMatch = symbol.symbolImportPattern.match(str(line))
        #         if declMatch:
        #             self.symbolsOnCurrentLine.append(symbol)
        #             self.processSymbolDeclMatch(declMatch, symbol)
        #             loadAsAdditionalModuleSymbol(symbol)
        #         elif importMatch:
        #             self.importsOnCurrentLine.append(symbol)
        #             self.processSymbolImportMatch(importMatch, symbol)
        #             loadAsAdditionalModuleSymbol(symbol)

        #validate the symbols on the current declaration line: Do they match the requirements for Hybrid Fortran?
        lineDeclarationType = DeclarationType.UNDEFINED
        arrayDeclarationLine = None
        for symbol in self.symbolsOnCurrentLine:
            if symbol.isArray and arrayDeclarationLine == False or not symbol.isArray and arrayDeclarationLine == True:
                raise UsageError(
                    "Array symbols have been mixed with non-array symbols on the same line (%s has declaration type %i). This is invalid in Hybrid Fortran. Please move apart these declarations.\n" %(
                        symbol.name,
                        symbol.declarationType
                    )
                )
            arrayDeclarationLine = symbol.isArray

    def processSymbolDeclMatch(self, paramDeclMatch, symbol):
        '''process everything that happens per h90 declaration symbol'''
        symbol.isMatched = True
        symbol.loadDeclaration(
            paramDeclMatch,
            self.patterns,
            self.currArguments if isinstance(self.currArguments, list) else []
        )

    def processSymbolImportMatch(self, importMatch, symbol):
        symbol.isMatched = True
        moduleName = importMatch.group(1)
        moduleNode = self.moduleNodesByName.get(moduleName)
        symbol.loadImportInformation(importMatch, self.cgDoc, moduleNode)

    def processInsideModuleState(self, line):
        super(H90CallGraphAndSymbolDeclarationsParser, self).processInsideModuleState(line)
        self.analyseSymbolInformationOnCurrentLine(line, isModuleSpecification=True)

    def processInsideDeclarationsState(self, line):
        '''process everything that happens per h90 declaration line'''
        super(H90CallGraphAndSymbolDeclarationsParser, self).processInsideDeclarationsState(line)
        if (self.state != 'inside_branch' and self.state != 'inside_declarations') or (self.state == "inside_branch" and self.stateBeforeBranch != "inside_declarations"):
            return
        self.analyseSymbolInformationOnCurrentLine(line)

    def processImplicitForeignModuleSymbolMatch(self, importMatch):
        pass

    def processModuleBeginMatch(self, moduleBeginMatch):
        super(H90CallGraphAndSymbolDeclarationsParser, self).processModuleBeginMatch(moduleBeginMatch)
        moduleName = moduleBeginMatch.group(1)
        moduleNode = self.moduleNodesByName.get(moduleName)
        if not moduleNode:
            return
        self.loadSymbolsFromTemplate(moduleNode, None, isModuleSymbols=True)

        #$$$ remove this in case we never enable routine domain dependant specifications for module symbols (likely)
        #just in case a programmer (like myself when doing the original ASUCA HF physics when HF didn't yet have module capabilities) specified module symbols within routine nodes instead of the module :-S.
        # self.tentativeModuleSymbolsByName = {}
        # for routineNode in self.routineNodesByModule.get(moduleName, []):
        #     routineDependantsByName = getSymbolsByName(
        #         self.cgDoc,
        #         routineNode,
        #         currentSymbolsByName=self.currSymbolsByName
        #     )
        #     for dependantName in routineDependantsByName.keys():
        #         symbol = routineDependantsByName[dependantName]
        #         if symbol.isArgument:
        #             continue
        #         self.tentativeModuleSymbolsByName[dependantName] = symbol

    def processModuleEndMatch(self, moduleEndMatch):
        super(H90CallGraphAndSymbolDeclarationsParser, self).processProcEndMatch(moduleEndMatch)
        dependants = self.currSymbolsByName.keys()
        unmatched = []
        for dependant in dependants:
            if not self.currSymbolsByName[dependant].isModuleSymbol:
                raise UsageError("Dependant %s has been referenced in a domain dependant region inside a module, but has never been matched." %(dependant))
            if self.currSymbolsByName[dependant].isMatched:
                continue
            unmatched.append(dependant)
        if len(unmatched) != 0:
            raise UsageError("The following non-scalar domain dependant declarations could not be found within module %s: %s;\n\
                domains of first unmatched: %s"
                %(self.currModuleName, unmatched, str(self.currSymbolsByName[unmatched[0]].domains))
            )
        logging.debug("Clearing current symbol scope since the module definition is finished", extra={"hfLineNo":currLineNo, "hfFile":currFile})
        self.currSymbolsByName = {}
        #$$$ remove this in case we never enable routine domain dependant specifications for module symbols (likely)
        # self.tentativeModuleSymbolsByName = None

    def processProcBeginMatch(self, subProcBeginMatch):
        super(H90CallGraphAndSymbolDeclarationsParser, self).processProcBeginMatch(subProcBeginMatch)
        subprocName = subProcBeginMatch.group(1)
        routineNode = self.routineNodesByProcName.get(subprocName)
        if not routineNode:
            return
        parallelRegionTemplates = self.parallelRegionTemplatesByProcName.get(self.currSubprocName)
        self.loadSymbolsFromTemplate(routineNode, parallelRegionTemplates)

    def processProcEndMatch(self, subProcEndMatch):
        super(H90CallGraphAndSymbolDeclarationsParser, self).processProcEndMatch(subProcEndMatch)
        routineNode = self.routineNodesByProcName.get(self.currSubprocName)
        dependants = self.currSymbolsByName.keys()
        unmatched = []
        for dependant in dependants:
            if self.currSymbolsByName[dependant].isModuleSymbol:
                continue
            if self.currSymbolsByName[dependant].isMatched or (routineNode and routineNode.getAttribute('parallelRegionPosition') in [None, '']):
                del self.currSymbolsByName[dependant]
                continue
            if len(self.currSymbolsByName[dependant].domains) == 0:
                #scalars that haven't been declared: Assume that they're from the local module
                #$$$ this code can probably be left away now that we analyze additional module symbols that haven't been declared domain dependant specifically within the module
                self.currSymbolsByName[dependant].sourceModule = "HF90_LOCAL_MODULE"
                self.currSymbolsByName[dependant].isModuleSymbol = True
                del self.currSymbolsByName[dependant]
                continue
            unmatched.append(dependant)
        if not routineNode:
            return
        if routineNode.getAttribute('parallelRegionPosition') in [None, '']:
            return
        #for routines that are 'grey' (no parallel region position) we want to unload symbols but not complain about unmatched stuff.
        if len(unmatched) != 0:
            raise Exception("The following non-scalar domain dependant declarations could not be found within subroutine %s: %s;\n\
                domains of first unmatched: %s"
                %(self.currSubprocName, unmatched, str(self.currSymbolsByName[unmatched[0]].domains))
            )

    def processLine(self, line):
        super(H90CallGraphAndSymbolDeclarationsParser, self).processLine(line)
        self.symbolsOnCurrentLine = []
        self.importsOnCurrentLine = []

class H90XMLSymbolDeclarationExtractor(H90CallGraphAndSymbolDeclarationsParser):
    entryNodesBySymbolName = {}
    currSymbols = []
    symbolsByModuleNameAndSymbolName = None

    def __init__(self, cgDoc, symbolsByModuleNameAndSymbolName=None, implementationsByTemplateName=None):
        super(H90XMLSymbolDeclarationExtractor, self).__init__(cgDoc, implementationsByTemplateName=implementationsByTemplateName)
        self.symbolsByModuleNameAndSymbolName = symbolsByModuleNameAndSymbolName

    def processSymbolAttributes(self, isModule=False):
        currSymbolNames = self.currSymbolsByName.keys()
        self.currSymbols = []
        if len(currSymbolNames) == 0:
            return
        self.currSymbols = [
            self.currSymbolsByName[symbolName]
            for symbolName in currSymbolNames
            if self.currSymbolsByName[symbolName].isModuleSymbol == isModule
        ]
        if len(self.currSymbols) == 0:
            return
        currParentName = self.currSubprocName if not isModule else self.currModuleName
        parentNode = self.routineNodesByProcName[currParentName] if not isModule else self.moduleNodesByName[currParentName]
        domainDependantRelationNodes = parentNode.getElementsByTagName("domainDependants")
        if domainDependantRelationNodes == None or len(domainDependantRelationNodes) == 0:
            raise Exception("we have active symbols (%s) loaded in %s but no domain dependant relation node can be found" %(
                self.currSymbols, currParentName
            ))
        domainDependantsRelationNode = domainDependantRelationNodes[0]
        domainDependantEntryNodes = domainDependantsRelationNode.getElementsByTagName("entry")
        if domainDependantEntryNodes == None or len(domainDependantEntryNodes) == 0:
            raise Exception("we have active symbols (%s) loaded in %s but no entry node can be found" %(
                self.currSymbols, currParentName
            ))
        self.entryNodesBySymbolName = {}
        for domainDependantEntryNode in domainDependantEntryNodes:
            self.entryNodesBySymbolName[domainDependantEntryNode.firstChild.nodeValue.strip()] = domainDependantEntryNode
        for symbol in self.currSymbols:
            entryNode = self.entryNodesBySymbolName.get(symbol.name)
            if entryNode:
                continue
            raise Exception("symbol %s is active but no information has been found in the codebase meta information" %(symbol))

    def storeCurrentSymbolAttributes(self, isModule=False):
        #store our symbol informations to the xml
        for symbol in self.currSymbols:
            if symbol.isModuleSymbol and isModule == False:
                continue
            entryNode = self.entryNodesBySymbolName.get(symbol.name)
            if not entryNode:
                continue
            symbol.storeDomainDependantEntryNodeAttributes(entryNode)

    def processImplicitForeignModuleSymbolMatch(self, importMatch):
        super(H90XMLSymbolDeclarationExtractor, self).processImplicitForeignModuleSymbolMatch(importMatch)
        if not self.symbolsByModuleNameAndSymbolName:
            return #in case we run this at a point where foreign symbol analysis is not available yet
        parentNode = None
        isInModuleScope = self.currSubprocName in [None, ""]
        if not isInModuleScope:
            parentNode = self.routineNodesByProcName.get(self.currSubprocName)
        else:
            parentNode = self.moduleNodesByName[self.currModuleName]
        parallelRegionPosition = parentNode.getAttribute("parallelRegionPosition")
        moduleName = importMatch.group(1)
        moduleSymbolParsingRequired = not self.implementation.supportsNativeModuleImportsWithinKernels and parallelRegionPosition in ["within", "outside"]
        moduleSymbolsByName = self.symbolsByModuleNameAndSymbolName.get(moduleName)
        if not moduleSymbolsByName and moduleSymbolParsingRequired:
            raise UsageError(
                "No symbol information for module %s. Please make this module available to Hybrid Fortran by moving it to a .h90 (.H90) file and use @domainDependant{attribute(host)} directives to declare the module symbols." %(
                    moduleName
                )
            )
        if not moduleSymbolsByName:
            return
        if moduleName == "":
            raise UsageError("import without module specified")
        symbolList = importMatch.group(2).split(',')
        for entry in symbolList:
            stripped = entry.strip()
            mappedImportMatch = self.patterns.singleMappedImportPattern.match(stripped)
            sourceSymbol = None
            symbolInScope = None
            if mappedImportMatch:
                symbolInScope = mappedImportMatch.group(1)
                sourceSymbol = mappedImportMatch.group(2)
            else:
                symbolInScope = stripped
                sourceSymbol = symbolInScope
            moduleSymbol = moduleSymbolsByName.get(uniqueIdentifier(sourceSymbol, moduleName))
            if not moduleSymbol and moduleSymbolParsingRequired:
                raise UsageError(
                    "No symbol information for symbol %s in module %s. Please make Hybrid Fortran aware of this symbol by declaring it in a @domainDependant{attribute(host)} directive in the module specification part." %(
                        sourceSymbol,
                        moduleName
                    )
                )
            elif not moduleSymbol:
                continue
            relationNode, templateNode = setTemplateInfos(
                self.cgDoc,
                parentNode,
                specText="attribute(autoDom)",
                templateParentNodeName="domainDependantTemplates",
                templateNodeName="domainDependantTemplate",
                referenceParentNodeName="domainDependants"
            )
            entries = addAndGetEntries(self.cgDoc, relationNode, symbolInScope)
            if len(entries) != 1:
                raise Exception("Could not add entry for symbol %s" %(entry))
            symbol = ImplicitForeignModuleSymbol(moduleName, symbolInScope, sourceSymbol, template=templateNode)
            symbol.isMatched = True
            symbol.loadDomainDependantEntryNodeAttributes(entries[0])
            if isInModuleScope:
                symbol.loadModuleNodeAttributes(parentNode)
            else:
                symbol.loadRoutineNodeAttributes(parentNode, self.parallelRegionTemplatesByProcName.get(self.currSubprocName))
            symbol.merge(moduleSymbol)
            if isInModuleScope:
                symbol.isModuleSymbol = True
                symbol.isHostSymbol = True
            else:
                symbol.isModuleSymbol = False
            self.currSymbolsByName[symbol.name] = symbol

    def processModuleEndMatch(self, moduleEndMatch):
        #get handles to currently active symbols -> temporarily save the handles
        self.processSymbolAttributes(isModule=True)
        logging.debug("exiting module %s. Storing informations for symbols %s" %(self.currModuleName, str(self.currSymbols)), extra={"hfLineNo":currLineNo, "hfFile":currFile})
        #finish parsing -> superclass destroys handles
        super(H90XMLSymbolDeclarationExtractor, self).processModuleEndMatch(moduleEndMatch)
        #store our symbol informations to the xml
        self.storeCurrentSymbolAttributes(isModule=True)
        #throw away our handles
        self.entryNodesBySymbolName = {}
        self.currSymbols = []


    def processProcEndMatch(self, subProcEndMatch):
        #get handles to currently active symbols -> temporarily save the handles
        self.processSymbolAttributes()
        logging.debug("exiting procedure %s. Storing informations for symbols %s" %(self.currSubprocName, str(self.currSymbols)), extra={"hfLineNo":currLineNo, "hfFile":currFile})
        #finish parsing -> superclass destroys handles
        super(H90XMLSymbolDeclarationExtractor, self).processProcEndMatch(subProcEndMatch)
        #store our symbol informations to the xml
        self.storeCurrentSymbolAttributes()
        #throw away our handles
        self.entryNodesBySymbolName = {}
        self.currSymbols = []