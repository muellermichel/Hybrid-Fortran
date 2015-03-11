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

##################### MAIN ##############################
#get all program arguments
parser = OptionParser()
parser.add_option("-i", "--sourceXML", dest="source",
                  help="read callgraph from this XML file", metavar="XML")
parser.add_option("-o", "--outFile", dest="output",
                  help="output png to OUT", metavar="OUT")
parser.add_option("-d", "--debug", action="store_true", dest="debug",
                  help="show debug print in standard error output")

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

graph = pydot.Dot(graph_type='digraph', rankdir='LR')

parallelRegions = doc.getElementsByTagName("activeParallelRegions")
routinesWithActiveRegions = []
for parallelRegion in parallelRegions:
  	if parallelRegion.parentNode.getAttribute("parallelRegionPosition") == "within":
  		routinesWithActiveRegions.append(parallelRegion.parentNode)

calls = doc.getElementsByTagName("call")
callGraphDict = {}
routineDict = {}
for routine in doc.getElementsByTagName("routine"):
	routineName = routine.getAttribute("name")
	addCallers(callGraphDict, routineDict, calls, routineName)
	addCallees(callGraphDict, routineDict, calls, routineName)

# for routine in routinesWithActiveRegions:
# 	routineName = routine.getAttribute("name")
# 	addCallers(callGraphDict, routineDict, calls, routineName)
# 	addCallees(callGraphDict, routineDict, calls, routineName)

callGraphEdges = callGraphDict.keys()
for callGraphEdge in callGraphEdges:
	edge = pydot.Edge(callGraphEdge[0], callGraphEdge[1])
	graph.add_edge(edge)

routines = doc.getElementsByTagName("routine")
routineNames = routineDict.keys()
for routineName in routineNames:
	routineMatch = None
	for routine in routines:
		if routine.getAttribute("name") == routineName:
			routineMatch = routine
			break
	if not routineMatch:
		continue
	regionPosition = routineMatch.getAttribute("parallelRegionPosition")
	fillColor = "gray"
	if regionPosition == "inside":
		fillColor = "green"
	elif regionPosition == "within":
		fillColor = "orange"
	elif regionPosition == "outside":
		fillColor = "red"

	node = pydot.Node(routineName, style="filled", fillcolor=fillColor)
	graph.add_node(node)

legend = pydot.Cluster(graph_name = 'Legend', label = 'Legend')
legend.add_node(pydot.Node(name='parallel region inside', style='filled', fillcolor="green"))
legend.add_node(pydot.Node(name='parallel region within', style='filled', fillcolor="orange"))
legend.add_node(pydot.Node(name='parallel region outside', style='filled', fillcolor="red"))
legend.add_node(pydot.Node(name='not affected by parallel region', style='filled', fillcolor="gray"))
legend.add_node(pydot.Node(name='not in h90 file'))
graph.add_subgraph(legend)

graph.write_png(output)



