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

from GeneralHelper import openFile, prettyprint
from DomHelper import addCallers, addCallees, getRegionPosition
from H90SymbolDependencyGraphAnalyzer import SymbolDependencyAnalyzer
from xml.dom.minidom import Document, parseString
from optparse import OptionParser
import pydot
import os
import sys

##################### MAIN ##############################
#get all program arguments
parser = OptionParser()
parser.add_option("-i", "--sourceXML", dest="source",
                  help="read callgraph from this XML file", metavar="XML")
parser.add_option("-o", "--outFile", dest="output",
                  help="output png to OUT", metavar="OUT")
parser.add_option("-d", "--debug", action="store_true", dest="debug",
                  help="show debug print in standard error output")
parser.add_option("--symbolName", dest="symbolName",
                  help="symbol to generate a dependency graph for")
parser.add_option("--symbolGraphRootRoutine", dest="symbolGraphRootRoutine",
                  help="symbol's root routine to generate a dependency graph for")
parser.add_option("--less", action="store_true", dest="less",
                  help="omit white and grey routines")

(options, args) = parser.parse_args()

if (not options.source):
  	sys.stderr.write("sourceXML option is mandatory. Use '--help' for informations on how to use this module\n")
  	sys.exit(1)

if options.symbolName and not options.symbolGraphRootRoutine:
	sys.stderr.write("symbolGraphRootRoutine option is mandatory if you use symbolName\n")
  	sys.exit(1)

output = "./out.png"
if options.output:
	output = options.output

legendFontSize = "8.0"
defaultFontSize = "15.0"
graphPenWidth = 2
moduleClusterPenwidth = 1
firstLevelClusterPenwidth = 3

#read in working xml
srcFile = openFile(str(options.source),'r')
data = srcFile.read()
srcFile.close()
doc = parseString(data)

analyzer = SymbolDependencyAnalyzer(doc)
if options.debug:
	print "=== calls by calleename ==="
	prettyprint(analyzer.callsByCalleeName)
	print "=== calls by callsByCallerName ==="
	prettyprint(analyzer.callsByCallerName)
	print "=== callGraphEdgesByCallerName ==="
	prettyprint(analyzer.callGraphEdgesByCallerName)

graph = pydot.Dot(graph_type='digraph', rankdir='LR', fontsize=defaultFontSize, compound=True)
# graph = pydot.Dot(graph_type='digraph', fontsize="27") #more useful for academic papers when there is a low number of nodes

### Callgraph Generation ###
callgraph = pydot.Cluster(graph_name = 'Callgraph', label = 'Callgraph', penwidth=firstLevelClusterPenwidth)
graph.add_subgraph(callgraph)
routineByName = {}
routinesBySourceDict = {}
sourceNameByRoutineNames = {}
routines = doc.getElementsByTagName("routine")
for routine in routines:
	routineName = routine.getAttribute("name")
	sourceName = routine.getAttribute("source")
	routinesInSourceList = routinesBySourceDict.get(sourceName, [])
	routinesInSourceList.append(routine)
	routinesBySourceDict[sourceName] = routinesInSourceList
	sourceNameByRoutineNames[routineName] = sourceName

sourceClustersByName = {}
for sourceName in routinesBySourceDict.keys():
	source = pydot.Cluster(sourceName, label=sourceName, penwidth=moduleClusterPenwidth)
	sourceClustersByName[sourceName] = source
	callgraph.add_subgraph(source)

aliasNamesByRoutineName = None
if options.symbolName:
	aliasNamesByRoutineName = analyzer.getAliasNamesByRoutineName(options.symbolName, options.symbolGraphRootRoutine)

for sourceName in sourceClustersByName.keys():
	source = sourceClustersByName[sourceName]
	for routine in routinesBySourceDict[sourceName]:
		routineName = routine.getAttribute("name")
		routineByName[routineName] = routine
		regionPosition = getRegionPosition(routineName, routines)
		fillColor = "gray"
		if regionPosition == "inside":
			fillColor = "green"
		elif regionPosition == "within":
			fillColor = "orange"
		elif regionPosition == "outside":
			fillColor = "red"
		label = routineName
		if aliasNamesByRoutineName:
			label = "%s s.alias: %s" %(routineName, aliasNamesByRoutineName.get(routineName, "n/a"))
		if options.less != True or fillColor != "gray":
			node = pydot.Node(routineName, label=label, style="filled", fillcolor=fillColor, fontsize=defaultFontSize, penwidth=graphPenWidth)
			source.add_node(node)

for callerName in analyzer.callGraphEdgesByCallerName.keys():
	for (caller, callee) in analyzer.callGraphEdgesByCallerName[callerName]:
		callerSourceName = sourceNameByRoutineNames[caller]
		calleeSourceName = sourceNameByRoutineNames[callee]
		regionPosition0 = getRegionPosition(caller, routines)
		if options.less == True and regionPosition0 not in ["inside", "within", "outside"]:
			continue
		regionPosition1 = getRegionPosition(callee, routines)
		if options.less == True and regionPosition1 not in ["inside", "within", "outside"]:
			continue
		edge = pydot.Edge(
			caller,
			callee,
			penwidth=graphPenWidth
		)
		callgraph.add_edge(edge)

legend = pydot.Cluster(graph_name = 'Legend', label = 'Legend', penwidth=moduleClusterPenwidth)
legend.add_node(pydot.Node(name='parallel region inside', style='filled', fillcolor="green", fontsize=legendFontSize, penwidth=graphPenWidth))
legend.add_node(pydot.Node(name='parallel region within', style='filled', fillcolor="orange", fontsize=legendFontSize, penwidth=graphPenWidth))
legend.add_node(pydot.Node(name='parallel region outside', style='filled', fillcolor="red", fontsize=legendFontSize, penwidth=graphPenWidth))
legend.add_node(pydot.Node(name='not affected by parallel region', style='filled', fillcolor="gray", fontsize=legendFontSize, penwidth=graphPenWidth))
legend.add_node(pydot.Node(name='not in h90 file', fontsize=legendFontSize, penwidth=graphPenWidth))
callgraph.add_subgraph(legend)

graph.write_png(output)