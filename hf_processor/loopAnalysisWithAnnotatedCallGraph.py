#!/usr/bin/python
# -*- coding: UTF-8 -*-

# Copyright (C) 2013 Michel Müller (Typhoon Computing), RIKEN Advanced Institute for Computational Science (AICS)

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
#  Procedure        loopAnalysisWithAnnotatedCallGraph.py              #
#  Comment          Analyses parallel regions within callgraph         #
#                   and adds information to dom respectively           #
#  Date             2012/07/27                                         #
#  Author           Michel Müller (AOKI Laboratory)                    #
#**********************************************************************#


from xml.dom.minidom import Document, parseString
from DomHelper import firstDuplicateChild, getNodeValue, getCalleesByCallerName, getCallersByCalleeName
from GeneralHelper import openFile
from optparse import OptionParser
import os
import sys
import pdb
import traceback

def getTemplateRelations(routineNode):
	templateRelations = []
	parallelRegionParents = routineNode.getElementsByTagName("parallelRegions")
	if not parallelRegionParents or len(parallelRegionParents) == 0:
		parallelRegionParents = routineNode.getElementsByTagName("activeParallelRegions")
	if parallelRegionParents and len(parallelRegionParents) > 0:
		templateRelations = parallelRegionParents[0].getElementsByTagName("templateRelation")
	return templateRelations

def addTemplateRelation(routineNode, templateRelation):
	parallelRegionsNodes = routineNode.getElementsByTagName("activeParallelRegions")
	parallelRegionNode = None
	if len(parallelRegionsNodes) == 0:
		parallelRegionNode = doc.createElement("activeParallelRegions")
		routineNode.appendChild(parallelRegionNode)
	else:
		parallelRegionNode = parallelRegionsNodes[0]
	templateID = templateRelation.getAttribute("id")
	newTemplateRelationNode = doc.createElement("templateRelation")
	newTemplateRelationNode.setAttribute("id", templateID)
	if (not firstDuplicateChild(parallelRegionNode, newTemplateRelationNode)):
		parallelRegionNode.appendChild(newTemplateRelationNode)

def addAttributeToAllCallGraphAncestors(routines, callNodesByCalleeName, routineNode, attributeName, attributeValue):
	routineName = routineNode.getAttribute("name")
	templateRelations = getTemplateRelations(routineNode)
	calls = callNodesByCalleeName.get(routineName)
	if not calls:
		return
	for call in calls:
		if call.getAttribute("callee") != routineName:
			raise Exception("Unexpected error: Wrong initialisation of caller-by-callee index.")
		#we got a match. search the caller routine
		callerName = call.getAttribute("caller")
		for routine in routines:
			if routine.getAttribute("name") == callerName:
				callerNode = routine
				routine.setAttribute(attributeName, attributeValue)
				for templateRelation in templateRelations:
					addTemplateRelation(routine, templateRelation)
				addAttributeToAllCallGraphAncestors(routines, callNodesByCalleeName, callerNode, attributeName, attributeValue)
				break


def addAttributeToAllCallGraphHeirs(routines, callNodesByCallerName, routineNode, attributeName, attributeValue):
	routineName = routineNode.getAttribute("name")
	parallelRegionPosition = routineNode.getAttribute("parallelRegionPosition")
	templateRelations = getTemplateRelations(routineNode)
	calls = callNodesByCallerName.get(routineName)
	if not calls:
		return
	for call in calls:
		if call.getAttribute("caller") != routineName:
			raise Exception("Unexpected error: Wrong initialisation of callee-by-caller index.")
		#check whether this call is within a parallel region. Only take into consideration those that are.
		if parallelRegionPosition == "within" and call.getAttribute("parallelRegionPosition") != "surround":
			continue
		#we got a match. search the caller routine
		calleeName = call.getAttribute("callee")
		for routine in routines:
			if routine.getAttribute("name") == calleeName:
				calleeNode = routine
				routine.setAttribute(attributeName, attributeValue)
				for templateRelation in templateRelations:
					addTemplateRelation(routine, templateRelation)
				addAttributeToAllCallGraphHeirs(routines, callNodesByCallerName, calleeNode, attributeName, attributeValue)
				break

#returns the first kernel caller that's being found in the calls by routine with name 'routineName'
def getFirstKernelCallerInCalleesOf(routineName, callNodesByCallerName, parallelRegionNodesByRoutineName):
	calls = callNodesByCallerName.get(routineName)
	if calls == None:
		return None
	for call in calls:
		calleeName = call.getAttribute("callee")
		kernelCallee = None
		subCalls = callNodesByCallerName.get(calleeName)
		if subCalls == None:
			continue
		for subCall in subCalls:
			if subCall.getAttribute("caller") != calleeName:
				raise Exception("Unexpected error: Wrong initialisation of callee-by-caller index.")
			if parallelRegionNodesByRoutineName.get(subCall.getAttribute("callee")) != None:
				kernelCallee = subCall
				break
		if kernelCallee != None:
			return call
	return None

def purgeTemplateRelation(routineNode, regionsNode, templateRelation):
	regionsNode.removeChild(templateRelation)
	remainingTemplateRelations = regionsNode.getElementsByTagName("parallelRegions")
	if not remainingTemplateRelations or len(remainingTemplateRelations) == 0:
		routineNode.removeChild(regionsNode)

def filterParallelRegionNodes(doc, routineNode, appliesTo):
	templates = doc.getElementsByTagName("parallelRegionTemplate")
	regionsNodes = routineNode.getElementsByTagName("parallelRegions")
	if not regionsNodes or len(regionsNodes) == 0:
		return
	regionsNode = regionsNodes[0]
	filtered = []
	templateRelations = regionsNode.getElementsByTagName("templateRelation")
	for templateRelation in templateRelations:
		templateID = templateRelation.getAttribute("id")
		matchedTemplate = None
		for template in templates:
			if template.getAttribute("id") == templateID:
				matchedTemplate = template
				break
		if matchedTemplate == None:
			raise Exception("Parallel region template id %s cannot be matched\n" %(templateID))
		appliesToNodes = matchedTemplate.getElementsByTagName("appliesTo")
		if (not appliesToNodes or len(appliesToNodes) == 0):
			if appliesTo != "":
				purgeTemplateRelation(routineNode, regionsNode, templateRelation)
				continue
		else:
			entries = appliesToNodes[0].getElementsByTagName("entry")
			for entry in entries:
				val = getNodeValue(entry).upper()
				if (val == "CPU" or val == "") and appliesTo == "":
					break
				if val == appliesTo.upper():
					break
			else:
				#inner loop never breaked -> no match found -> purge this relation
				purgeTemplateRelation(routineNode, regionsNode, templateRelation)
				continue

def analyseParallelRegions(doc, appliesTo):
	callNodes = doc.getElementsByTagName("call")
	routineNodes = doc.getElementsByTagName("routine")
	for routineNode in routineNodes:
		filterParallelRegionNodes(doc, routineNode, appliesTo)
	callNodesByCallerName = getCalleesByCallerName(callNodes)
	callNodesByCalleeName = getCallersByCalleeName(callNodes)

	parallelRegionNodes = doc.getElementsByTagName("parallelRegions")
	parallelRegionNodesByRoutineName = {}
	for parallelRegionNode in parallelRegionNodes:
		routine = parallelRegionNode.parentNode
		routineName = routine.getAttribute("name")
		if appliesTo == "GPU" and parallelRegionNodesByRoutineName.get(routineName) != None:
			raise Exception("Multiple GPU parallel regions in subroutine %s" %(routineName))
		parallelRegionNodesByRoutineName[routineName] = parallelRegionNode

	kernelCallerProblemFound = False
	messagesPresentedFor = []
	for parallelRegionNode in parallelRegionNodes:
		if not parallelRegionNode.parentNode.tagName == "routine":
			raise Exception("Parallel region not within routine.")

		parallelRegionNode.parentNode.setAttribute("parallelRegionPosition", "within")
		routine = parallelRegionNode.parentNode
		routineName = routine.getAttribute("name")
		if routineName == None:
			raise Exception("Unexpected error: Kernel routine without name")
		addAttributeToAllCallGraphAncestors(routineNodes, callNodesByCalleeName, routine, "parallelRegionPosition", "inside")
		addAttributeToAllCallGraphHeirs(routineNodes, callNodesByCallerName, routine, "parallelRegionPosition", "outside")

		#rename this parallelRegion node to 'activeParallelRegion'
		children = parallelRegionNode.childNodes
		newRegionNode = doc.createElement("activeParallelRegions")
		for child in children:
			newRegionNode.appendChild(child)
		routine.removeChild(parallelRegionNode)
		routine.appendChild(newRegionNode)

		if appliesTo != "GPU":
			continue

		#check our kernel callers. Rule: Their direct callees should not be a kernel caller themselves.
		kernelCallers = callNodesByCalleeName.get(routineName)
		if not kernelCallers:
			continue

		for kernelCaller in kernelCallers:
			kernelCallerName = kernelCaller.getAttribute("caller")
			kernelWrapperCall = getFirstKernelCallerInCalleesOf(kernelCallerName, callNodesByCallerName, parallelRegionNodesByRoutineName)
			if kernelWrapperCall != None:
				kernelWrapperName = kernelWrapperCall.getAttribute("callee")
				if not kernelCallerProblemFound:
					sys.stderr.write("WARNING: Subroutine %s calls at least one kernel (%s) and at least one kernel wrapper (%s). \
This may cause device attribute mismatch compiler errors. In this case please wrap all kernels called by %s, such that it does not call a mix of kernel wrappers and kernels.\n"
						%(kernelCallerName, routineName, kernelWrapperName, kernelCallerName)
					)
					kernelCallerProblemFound = True
					messagesPresentedFor.append(kernelCallerName)
				elif kernelCallerName not in messagesPresentedFor:
					messagesPresentedFor.append(kernelCallerName)
					sys.stderr.write("...same for %s: calls kernel %s, kernel wrapper %s\n" %(kernelCallerName, routineName, kernelWrapperName))

##################### MAIN ##############################
#get all program arguments
parser = OptionParser()
parser.add_option("-i", "--sourceXML", dest="source",
                  help="read callgraph from this XML file", metavar="XML")
parser.add_option("-a", "--appliesTo", dest="appliesTo",
                  help="specify the framework for which the loopstructure shall be extracted (as specified in the appliesTo section in parallelRegion definitions)")
parser.add_option("-d", "--debug", action="store_true", dest="debug",
                  help="show debug print in standard error output"
                  )
parser.add_option("-p", "--pretty", action="store_true", dest="pretty",
                  help="make xml output pretty")
(options, args) = parser.parse_args()

if (not options.source):
    sys.stderr.write("sourceXML option is mandatory. Use '--help' for informations on how to use this module\n")
    sys.exit(1)

appliesTo = ""
if options.appliesTo and options.appliesTo.upper() != "CPU":
	appliesTo = options.appliesTo

#read in working xml
srcFile = openFile(str(options.source),'r')
data = srcFile.read()
srcFile.close()
doc = parseString(data)

try:
	analyseParallelRegions(doc, appliesTo)
except Exception, e:
	sys.stderr.write('Error when analysing callgraph file file %s: %s\n'
		%(str(options.source), str(e))
	)
	if options.debug:
		sys.stderr.write(traceback.format_exc())
	sys.exit(1)

if (options.pretty):
	sys.stdout.write(doc.toprettyxml())
else:
	sys.stdout.write(doc.toxml())

