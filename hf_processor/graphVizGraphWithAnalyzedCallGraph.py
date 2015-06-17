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
from H90SymbolDependencyGraphAnalyzer import SymbolDependencyAnalyzer, SymbolType, SymbolAnalysis
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
parser.add_option("--allSymbols", action="store_true", dest="allSymbols",
                  help="show table of all symbols passed through subroutine in tables for each subroutine")
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

smallFontSize = "8.0"
defaultFontSize = "10.0"
clusterFontSize = "30.0"
graphPenWidth = 1
moduleClusterPenwidth = 1
firstLevelClusterPenwidth = 3
font = "Sans-Serif"
symbolColorsByType = {
	SymbolType.UNDEFINED : "#ffa3a3",
    SymbolType.ARGUMENT_WITH_DOMAIN_DEPENDANT_SPEC : "#afffad",
    SymbolType.ARGUMENT : "#ffcd94",
    SymbolType.MODULE_DATA_USED_IN_CALLEE_GRAPH : "#c76060",
    SymbolType.MODULE_DATA : "#ff5747",
    SymbolType.MODULE_DATA_USED_IN_CALLEE_GRAPH_WITH_DOMAIN_DEPENDANT_SPEC : "#bf60c7",
    SymbolType.MODULE_DATA_WITH_DOMAIN_DEPENDANT_SPEC : "#00ccf0",
    SymbolType.DOMAIN_DEPENDANT : "#aac760"
}

def getNodeLabel(routineName, symbolAnalysis, regionPosition):
	symbolRows = []
	fillColor = "gray"
	if regionPosition == "inside":
		fillColor = "green"
	elif regionPosition == "within":
		fillColor = "orange"
	elif regionPosition == "outside":
		fillColor = "red"
	for analysisEntry in symbolAnalysis:
		symbolRowColor = symbolColorsByType[analysisEntry.symbolType]
		symbolRows.append("<TD BGCOLOR='%s'>%s</TD>" %(
			symbolRowColor,
			("</TD><TD BGCOLOR='%s'>" %(symbolRowColor)).join([
				"<FONT POINT-SIZE='%s'>%s</FONT>" %(smallFontSize, entry) if entry else ""
				for entry in [
					str(analysisEntry.aliasNamesByRoutineName.get(routineName, analysisEntry.name)),
					str(analysisEntry.sourceSymbol),
					str(analysisEntry.sourceModule)
				]
			])
		))
	return "<<TABLE BORDER='0' CELLBORDER='1' CELLSPACING='0' CELLPADDING='4'><TR>%s</TR></TABLE>>" %(
		"</TR><TR>".join([ #routine name header
					"<TD COLSPAN='3' BGCOLOR='%s'>%s</TD>" %(fillColor, routineName)
				] +
				# [ #labels
				# 	"<TD>%s</TD>" %("</TD><TD>".join([
				# 		"<FONT POINT-SIZE='%s'>%s</FONT>" %(smallFontSize, entry) for entry in ["local", "s.name", "s.module"] #
				# 	]))
				# ] +
				symbolRows #symbols
		)
	)

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
	print "=== routines by name ==="
	prettyprint(analyzer.routinesByName)

analysis = None
if options.symbolGraphRootRoutine:
	analysis = analyzer.getSymbolAnalysisByRoutine(options.symbolGraphRootRoutine)
elif options.allSymbols:
	analysis = analyzer.getSymbolAnalysisByRoutine()
if options.debug and analysis != None:
	print "=== analysis ==="
	prettyprint(analysis)

if analysis == None:
	#without the symbol analysis the graph looks best left-to-right
	graph = pydot.Dot(graph_type='digraph', rankdir='LR', fontsize=defaultFontSize, compound=True)
else:
	graph = pydot.Dot(graph_type='digraph', fontsize=defaultFontSize, compound=True)
# graph = pydot.Dot(graph_type='digraph', fontsize="27") #more useful for academic papers when there is a low number of nodes

### Callgraph Generation ###
# callgraph = pydot.Cluster(graph_name = 'Callgraph', label = 'Callgraph', fontname=font, penwidth=firstLevelClusterPenwidth)
# graph.add_subgraph(callgraph)
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
	source = pydot.Cluster(sourceName, label=sourceName, fontsize=clusterFontSize, fontname=font, penwidth=moduleClusterPenwidth, fillcolor="#e1e1e0", style="rounded, filled")
	sourceClustersByName[sourceName] = source
	graph.add_subgraph(source)

aliasNamesByRoutineName = None
if options.symbolName:
	aliasNamesByRoutineName = analyzer.getAliasNamesByRoutineName(options.symbolName, options.symbolGraphRootRoutine)

for sourceName in sourceClustersByName.keys():
	source = sourceClustersByName[sourceName]
	for routine in routinesBySourceDict[sourceName]:
		routineName = routine.getAttribute("name")
		routineByName[routineName] = routine
		regionPosition = getRegionPosition(routineName, routines)
		symbolAnalysis = []
		if analysis != None:
			for symbolName in analysis.get(routineName, {}).keys():
				for callEntry in analysis[routineName][symbolName]:
					symbolAnalysis.append(callEntry)
		label = getNodeLabel(routineName, symbolAnalysis, regionPosition)
		if aliasNamesByRoutineName:
			label = "%s s.alias: %s" %(routineName, aliasNamesByRoutineName.get(routineName, "n/a"))
		if not options.less or regionPosition in ["inside", "within", "outside"]:
			node = pydot.Node(routineName, label=label, shape="plaintext", fontname=font, fontsize=defaultFontSize, penwidth=graphPenWidth)
			source.add_node(node)

edges = {}
for callerName in analyzer.callGraphEdgesByCallerName.keys():
	for (caller, callee) in analyzer.callGraphEdgesByCallerName[callerName]:
		edges[(caller, callee)] = None
for (caller, callee) in edges.keys():
	callerSourceName = sourceNameByRoutineNames.get(caller)
	calleeSourceName = sourceNameByRoutineNames.get(callee)
	if not callerSourceName or not calleeSourceName:
		continue
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
	graph.add_edge(edge)

legend = pydot.Cluster(graph_name = 'Legend', label = 'Legend', penwidth=moduleClusterPenwidth)
exampleSymbolAnalysis1 = SymbolAnalysis()
exampleSymbolAnalysis1.name = "local_name (module data with domain dependant spec)"
exampleSymbolAnalysis1.sourceModule = "source module"
exampleSymbolAnalysis1.sourceSymbol = "source symbol name"
exampleSymbolAnalysis1.symbolType = SymbolType.MODULE_DATA_WITH_DOMAIN_DEPENDANT_SPEC
exampleSymbolAnalysis2 = SymbolAnalysis()
exampleSymbolAnalysis2.name = "local_name (module data used in callee graph with domain dependant spec)"
exampleSymbolAnalysis2.sourceModule = "source module"
exampleSymbolAnalysis2.sourceSymbol = "source symbol name"
exampleSymbolAnalysis2.symbolType = SymbolType.MODULE_DATA_USED_IN_CALLEE_GRAPH_WITH_DOMAIN_DEPENDANT_SPEC
exampleSymbolAnalysis3 = SymbolAnalysis()
exampleSymbolAnalysis3.name = "local_name (module data without domain dependant spec)"
exampleSymbolAnalysis3.sourceModule = "source module"
exampleSymbolAnalysis3.sourceSymbol = "source symbol name"
exampleSymbolAnalysis3.symbolType = SymbolType.MODULE_DATA
exampleSymbolAnalysis4 = SymbolAnalysis()
exampleSymbolAnalysis4.name = "local_name (module data used in callee graph without domain dependant spec)"
exampleSymbolAnalysis4.sourceModule = "source module"
exampleSymbolAnalysis4.sourceSymbol = "source symbol name"
exampleSymbolAnalysis4.symbolType = SymbolType.MODULE_DATA_USED_IN_CALLEE_GRAPH
exampleSymbolAnalysis5 = SymbolAnalysis()
exampleSymbolAnalysis5.name = "local_name of routine argument with domain dependant spec"
exampleSymbolAnalysis5.symbolType = SymbolType.ARGUMENT_WITH_DOMAIN_DEPENDANT_SPEC
exampleSymbolAnalysis6 = SymbolAnalysis()
exampleSymbolAnalysis6.name = "local_name of routine argument without domain dependant spec"
exampleSymbolAnalysis6.symbolType = SymbolType.ARGUMENT
exampleSymbolAnalysis7 = SymbolAnalysis()
exampleSymbolAnalysis7.name = "name of local symbol with domain dependant spec"
exampleSymbolAnalysis7.symbolType = SymbolType.DOMAIN_DEPENDANT
exampleSymbolAnalysis8 = SymbolAnalysis()
exampleSymbolAnalysis8.name = "name of local symbol without domain dependant spec"
exampleSymbolAnalysis8.symbolType = SymbolType.UNDEFINED
exampleSymbolAnalysis = [
	exampleSymbolAnalysis1,
	exampleSymbolAnalysis2,
	exampleSymbolAnalysis3,
	exampleSymbolAnalysis4,
	exampleSymbolAnalysis5,
	exampleSymbolAnalysis6,
	exampleSymbolAnalysis7,
	exampleSymbolAnalysis8
]
legend.add_node(pydot.Node("example", label=getNodeLabel('routine name (parallel region inside)', exampleSymbolAnalysis, "inside"), shape="plaintext", fontname=font, fontsize=defaultFontSize, penwidth=graphPenWidth))
legend.add_node(pydot.Node("example2", label=getNodeLabel('routine name (parallel region within)', [], "within"), shape="plaintext", fontname=font, fontsize=defaultFontSize, penwidth=graphPenWidth))
legend.add_node(pydot.Node("example3", label=getNodeLabel('routine name (parallel region outside)', [], "outside"), shape="plaintext", fontname=font, fontsize=defaultFontSize, penwidth=graphPenWidth))
legend.add_node(pydot.Node("example4", label=getNodeLabel('routine name (not affected by parallel region)', [], ""), shape="plaintext", fontname=font, fontsize=defaultFontSize, penwidth=graphPenWidth))
graph.add_subgraph(legend)
graph.write_png(output)