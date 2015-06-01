#!/usr/bin/python
# -*- coding: UTF-8 -*-

# Copyright (C) 2015 Michel Müller, Tokyo Institute of Technology

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
#  Procedure        graphVizGraphWithAnalyzedCallGraph.py              #
#  Comment          Create a graphical representation                  #
#  Date             2012/07/27                                         #
#  Author           Michel Müller (AOKI Laboratory)                    #
#**********************************************************************#

from GeneralHelper import openFile
from xml.dom.minidom import Document, parseString
from optparse import OptionParser
import pydot
import os
import sys

def addCallers(callGraphDict, routineDict, calls, routineName):
	for call in calls:
		callee = call.getAttribute("callee")
		caller = call.getAttribute("caller")
		if callee == routineName:
			entry = (caller, callee)
			callGraphDict[entry] = ""
			routineDict[caller] = ""
			routineDict[callee] = ""
			addCallers(callGraphDict, routineDict, calls, caller)

def addCallees(callGraphDict, routineDict, calls, routineName):
	for call in calls:
		callee = call.getAttribute("callee")
		caller = call.getAttribute("caller")
		if caller == routineName:
			entry = (caller, callee)
			callGraphDict[entry] = ""
			routineDict[caller] = ""
			routineDict[callee] = ""
			addCallees(callGraphDict, routineDict, calls, callee)

def getRegionPosition(routineName):
	routineMatch = None
	for routine in routines:
		if routine.getAttribute("name") == routineName:
			routineMatch = routine
			break
	if not routineMatch:
		return None
	regionPosition = routineMatch.getAttribute("parallelRegionPosition")
	if regionPosition == None:
		return 'unspecified'
	return regionPosition

##################### MAIN ##############################
#get all program arguments
parser = OptionParser()
parser.add_option("-i", "--sourceXML", dest="source",
                  help="read callgraph from this XML file", metavar="XML")
parser.add_option("-o", "--outFile", dest="output",
                  help="output png to OUT", metavar="OUT")
parser.add_option("-d", "--debug", action="store_true", dest="debug",
                  help="show debug print in standard error output")
parser.add_option("--less", action="store_true", dest="less",
                  help="omit white and grey routines")

(options, args) = parser.parse_args()

if (not options.source):
  	sys.stderr.write("sourceXML option is mandatory. Use '--help' for informations on how to use this module\n")
  	sys.exit(1)

output = "./out.png"
if options.output:
	output = options.output

#read in working xml
srcFile = openFile(str(options.source),'r')
data = srcFile.read()
srcFile.close()
doc = parseString(data)

graph = pydot.Dot(graph_type='digraph', rankdir='LR', fontsize="20", compound=True)
# graph = pydot.Dot(graph_type='digraph', fontsize="27") #more useful for academic papers when there is a low number of nodes

parallelRegions = doc.getElementsByTagName("activeParallelRegions")
routinesWithActiveRegions = []
for parallelRegion in parallelRegions:
  	if parallelRegion.parentNode.getAttribute("parallelRegionPosition") == "within":
  		routinesWithActiveRegions.append(parallelRegion.parentNode)

calls = doc.getElementsByTagName("call")
callGraphDict = {}
routineDict = {}
routineByName = {}
routinesBySourceDict = {}
sourceByRoutineNames = {}
sourceNameByRoutineNames = {}
for routine in doc.getElementsByTagName("routine"):
	routineName = routine.getAttribute("name")
	sourceName = routine.getAttribute("source")
	routinesInSourceList = routinesBySourceDict.get(sourceName, [])
	routinesInSourceList.append(routine)
	routinesBySourceDict[sourceName] = routinesInSourceList
	sourceNameByRoutineNames[routineName] = sourceName
	addCallers(callGraphDict, routineDict, calls, routineName)
	addCallees(callGraphDict, routineDict, calls, routineName)

routines = doc.getElementsByTagName("routine")
routineNames = routineDict.keys()

sourceClustersByName = {}
for sourceName in routinesBySourceDict.keys():
	source = pydot.Cluster(sourceName, label=sourceName, penwidth=3)
	sourceClustersByName[sourceName] = source
	graph.add_subgraph(source)

for sourceName in sourceClustersByName.keys():
	source = sourceClustersByName[sourceName]
	for routine in routinesBySourceDict[sourceName]:
		routineName = routine.getAttribute("name")
		routineByName[routineName] = routine
		regionPosition = getRegionPosition(routineName)
		fillColor = "gray"
		if regionPosition == "inside":
			fillColor = "green"
		elif regionPosition == "within":
			fillColor = "orange"
		elif regionPosition == "outside":
			fillColor = "red"
		if options.less != True or fillColor != "gray":
			node = pydot.Node(routineName, style="filled", fillcolor=fillColor, fontsize="30.0", penwidth=5)
			source.add_node(node)

callGraphEdges = callGraphDict.keys()
for (caller, callee) in callGraphEdges:
	callerSourceName = sourceNameByRoutineNames[caller]
	calleeSourceName = sourceNameByRoutineNames[callee]
	regionPosition0 = getRegionPosition(caller)
	if options.less == True and regionPosition0 not in ["inside", "within", "outside"]:
		continue
	regionPosition1 = getRegionPosition(callee)
	if options.less == True and regionPosition1 not in ["inside", "within", "outside"]:
		continue
	edge = pydot.Edge(
		caller,
		callee,
		penwidth=5
		# ltail=sourceClustersByName[callerSourceName].get_name(),
		# lhead=sourceClustersByName[calleeSourceName].get_name()
	)
	graph.add_edge(edge)

legend = pydot.Cluster(graph_name = 'Legend', label = 'Legend', penwidth=3)
legend.add_node(pydot.Node(name='parallel region inside', style='filled', fillcolor="green", fontsize="30.0", penwidth=5))
legend.add_node(pydot.Node(name='parallel region within', style='filled', fillcolor="orange", fontsize="30.0", penwidth=5))
legend.add_node(pydot.Node(name='parallel region outside', style='filled', fillcolor="red", fontsize="30.0", penwidth=5))
legend.add_node(pydot.Node(name='not affected by parallel region', style='filled', fillcolor="gray", fontsize="30.0", penwidth=5))
legend.add_node(pydot.Node(name='not in h90 file', fontsize="30.0", penwidth=5))
graph.add_subgraph(legend)
graph.write_png(output)