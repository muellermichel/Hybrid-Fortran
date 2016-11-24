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

from tools.metadata import parseString, ImmutableDOMDocument, getClonedDocument, getParallelDomainNames
from optparse import OptionParser
from machinery.parser import H90XMLSymbolDeclarationExtractor, getModuleNodesByName, getParallelRegionData
from machinery.converter import H90toF90Converter, getSymbolsByRoutineNameAndSymbolName, getSymbolsByModuleNameAndSymbolName
from machinery.commons import conversionOptions, FortranCodeSanitizer
from tools.commons import UsageError, openFile, getDataFromFile, setupDeferredLogging, printProgressIndicator, progressIndicatorReset
from tools.filesystem import dirEntries
from tools.analysis import SymbolDependencyAnalyzer
import implementations.fortran
from io import FileIO
import os, errno, sys, json, traceback, logging

def print_cycles(objects, outstream=sys.stdout, show_progress=False):
	"""
	objects:       A list of objects to find cycles in.  It is often useful
				   to pass in gc.garbage to find the cycles that are
				   preventing some objects from being garbage collected.
	outstream:     The stream for output.
	show_progress: If True, print the number of objects reached as they are
				   found.
	"""

	import gc
	from types import FrameType

	def print_path(path):
		for i, step in enumerate(path):
			# next "wraps around"
			next = path[(i + 1) % len(path)]

			outstream.write("   %s -- " % str(type(step)))
			if isinstance(step, dict):
				for key, val in step.items():
					if val is next:
						outstream.write("[%s]" % repr(key))
						break
					if key is next:
						outstream.write("[key] = %s" % repr(val))
						break
			elif isinstance(step, list):
				outstream.write("[%d]" % step.index(next))
			elif isinstance(step, tuple):
				outstream.write("[%d]" % list(step).index(next))
			else:
				outstream.write(repr(step))
			outstream.write(" ->\n")
		outstream.write("\n")

	def recurse(obj, start, all, current_path):
		if show_progress:
			outstream.write("%d\r" % len(all))

		all[id(obj)] = None

		referents = gc.get_referents(obj)
		for referent in referents:
			# If we've found our way back to the start, this is
			# a cycle, so print it out
			if referent is start:
				print_path(current_path)

			# Don't go back through the original list of objects, or
			# through temporary references to the object, since those
			# are just an artifact of the cycle detector itself.
			elif referent is objects or isinstance(referent, FrameType):
				continue

			# We haven't seen this object before, so recurse
			elif id(referent) not in all:
				recurse(referent, start, all, current_path + [obj])

	for obj in objects:
		outstream.write("Examining: %r\n" % obj)
		recurse(obj, obj, { }, [])

def setDeviceHandlingFlagsInCallGraph(routine, calleesByCallerName, calleesByCalleeName, routinesByName, alreadyHandledRoutinesByName):
	alreadyHandledRoutines = alreadyHandledRoutinesByName.get(routine.name, [])
	for alreadyHandledRoutine in alreadyHandledRoutines:
		if routine is alreadyHandledRoutine:
			return
	alreadyHandledRoutines.append(routine)
	alreadyHandledRoutinesByName[routine.name] = alreadyHandledRoutines
	routine.isUsedInHostOnlyContext = True
	for calleeRoutineVersion in calleesByCalleeName.get(routine.name, []):
		calleeRoutineVersion.isUsedInHostOnlyContext = True
	for callee in calleesByCallerName.get(routine.name, []):
		calleeRoutine = routinesByName.get(callee.name)
		if calleeRoutine:
			calleeRoutine.isUsedInHostOnlyContext = True #duplicate the flag setting for the implementation routines
		setDeviceHandlingFlagsInCallGraph(callee, calleesByCallerName, calleesByCalleeName, routinesByName, alreadyHandledRoutinesByName)
		for indirectCallee in calleesByCalleeName.get(callee.name, []):
			setDeviceHandlingFlagsInCallGraph(indirectCallee, calleesByCallerName, calleesByCalleeName, routinesByName, alreadyHandledRoutinesByName)

def makeFileContent(cgDoc, filesInDir, implementationsByTemplateName):
	#   build up meta informations about the whole codebase
	symbolAnalysisByRoutineNameAndSymbolName = None
	symbolsByModuleNameAndSymbolName = None
	symbolsByRoutineNameAndSymbolName = None
	parallelDomainNames = None
	moduleNodesByName = None
	parallelRegionData = None
	try:
		sys.stderr.write('Processing informations about the whole codebase\n')
		moduleNodesByName = getModuleNodesByName(cgDoc)
		parallelRegionData = getParallelRegionData(cgDoc)
		symbolAnalyzer = SymbolDependencyAnalyzer(cgDoc)
		#next line writes some information to cgDoc as a sideeffect. $$$ clean this up, ideally make cgDoc immutable everywhere for better performance
		symbolAnalysisByRoutineNameAndSymbolName = symbolAnalyzer.getSymbolAnalysisByRoutine()
		symbolsByModuleNameAndSymbolName = getSymbolsByModuleNameAndSymbolName(
			ImmutableDOMDocument(cgDoc),
			moduleNodesByName,
			symbolAnalysisByRoutineNameAndSymbolName=symbolAnalysisByRoutineNameAndSymbolName
		)
		symbolsByRoutineNameAndSymbolName = getSymbolsByRoutineNameAndSymbolName(
			ImmutableDOMDocument(cgDoc),
			parallelRegionData[2],
			parallelRegionData[1],
			symbolAnalysisByRoutineNameAndSymbolName=symbolAnalysisByRoutineNameAndSymbolName
		)
		parallelDomainNames = getParallelDomainNames(cgDoc)
	except UsageError as e:
		logging.error('Error: %s' %(str(e)))
		sys.exit(1)
	except Exception as e:
		logging.critical('Error when processing meta information about the codebase: %s' %(str(e)))
		logging.info(traceback.format_exc())
		sys.exit(1)

	#   Prepare the content for all files based on all the information above.
	fileContents = []
	for fileNum, fileInDir in enumerate(filesInDir):
		printProgressIndicator(sys.stderr, fileInDir, fileNum + 1, len(filesInDir), "Preparing File Content")
		try:
			converter = H90toF90Converter(
				ImmutableDOMDocument(cgDoc), #using our immutable version we can speed up ALL THE THINGS through caching
				implementationsByTemplateName,
				moduleNodesByName,
				parallelRegionData,
				symbolAnalysisByRoutineNameAndSymbolName,
				symbolsByModuleNameAndSymbolName,
				symbolsByRoutineNameAndSymbolName,
				parallelDomainNames
			)
			fileContents.append(converter.prepareFileContent(fileInDir))
		except UsageError as e:
			logging.error('Error: %s' %(str(e)))
			sys.exit(1)
	progressIndicatorReset(sys.stderr)
	return fileContents

def makeCallGraphAnalysis(fileContents):
	#   Analyse Callgraph for Implementation specific behavior
	modulesByName = {}
	routinesByName = {}
	calleesByCallerName = {}
	calleesByCalleeName = {}
	for fileNum, fc in enumerate(fileContents):
		printProgressIndicator(sys.stderr, fc['fileName'], fileNum + 1, len(fileContents), "CG Analysis")
		for m in fc['modules']:
			if modulesByName.get(m.name) != None:
				logging.error("Error: Multiple modules with name %s found" %(m.name))
				sys.exit(1)
			modulesByName[m.name] = m
			for r in m.routines:
				if routinesByName.get(r.name) != None:
					logging.error("Error: Multiple routines with name %s found" %(r.name))
					sys.exit(1)
				routinesByName[r.name] = r
				for callee in r.callees:
					callees = calleesByCallerName.get(r.name, [])
					callees.append(callee)
					calleesByCallerName[r.name] = callees

					callees = calleesByCalleeName.get(callee.name, [])
					callees.append(callee)
					calleesByCalleeName[callee.name] = callees
	progressIndicatorReset(sys.stderr)

	#   Analyse Callgraph for device handling capabilities
	routines = routinesByName.values()
	alreadyHandledRoutinesByName = {}
	for routineNum, r in enumerate(routines):
		printProgressIndicator(sys.stderr, r.name, routineNum + 1, len(routines), "CG Device Handling Analysis")
		if not hasattr(r, "implementation"):
			continue
		if r.implementation.canHandleDeviceData and r.node.getAttribute('parallelRegionPosition') in [
			"within",
			"inside",
			"outside"
		]:
			continue
		setDeviceHandlingFlagsInCallGraph(r, calleesByCallerName, calleesByCalleeName, routinesByName, alreadyHandledRoutinesByName)
	progressIndicatorReset(sys.stderr)
	return modulesByName, routinesByName

def prepareImplementation(fileContents):
	#   Preprocess all modules.
	#   Routines will be split according to architecture.
	#   Symbol usage will be analysed so this info is available globally.
	for fileNum, fc in enumerate(fileContents):
		printProgressIndicator(sys.stderr, fc['fileName'], fileNum + 1, len(fileContents), "Prepare Modules for Implementation")
		try:
			for m in fc['modules']:
				m.prepareForImplementation()
		except UsageError as e:
			logging.error('Error: %s' %(str(e)))
			sys.exit(1)
	progressIndicatorReset(sys.stderr)

def writeOut(fileName, content):
	outputPath = os.path.join(
		os.path.normpath(options.outputDir),
		os.path.splitext(os.path.basename(fileName))[0] + ".P90.temp"
	)
	outputStream = FileIO(outputPath, mode="wb")
	try:
		outputStream.write(content)
	finally:
		outputStream.close()
	return fileName

#   Finally, do the conversion based on the prepare content
def implement(fileContent, modulesByName, routinesByName, sanitizer):
	return writeOut(
		fileContent["fileName"],
		sanitizer.sanitizeLines("%s\n%s" %(
			fileContent['prefix'],
			"".join([
				"%s\n\n%s\n" %(
					m.implemented(modulesByName, routinesByName),
					fileContent['appendixByModuleName'].get(m.name, "")
				)
				for m in fileContent['modules']
			])
		))
	)

def convertEverything(fileContents, modulesByName, routinesByName):
	import math
	import functools
	import multiprocessing as mp
	codeSanitizer = FortranCodeSanitizer()
	cpu_count = min(mp.cpu_count(), len(fileContents))
	# cpu_count = 1
	chunkSize = max(math.ceil(float(len(fileContents)) / cpu_count), 1)
	workerPool = mp.Pool(cpu_count)
	import pickle
	sys.setrecursionlimit(10000) #pickle fails without this line
	# for fc in fileContents:
	# 	sys.stderr.write("checking %s\n" %(fc["fileName"]))
	# 	# print_cycles([implement] + modulesByName.values() + routinesByName.values() + fileContents)
	# 	# sys.exit(1)
	# 	_ = pickle.loads(pickle.dumps(fc))
	sys.stderr.write("Starting %i workers, each getting %i sized chunks to convert our %i files\n" %(
		cpu_count,
		chunkSize,
		len(fileContents)
	))
	for fileNum, fileName in enumerate(workerPool.imap_unordered(
		functools.partial(implement, modulesByName=modulesByName, routinesByName=routinesByName, sanitizer=codeSanitizer),
		fileContents,
		chunksize=chunkSize
	)):
		printProgressIndicator(sys.stderr, fileName, fileNum + 1, len(fileContents), "Implementing as Standard Fortran")
	progressIndicatorReset(sys.stderr)

def generateCodbase(options):
	conversionOptions.debugPrint = options.debug
	filesInDir = dirEntries(str(options.sourceDir), True, 'h90')

	try:
		os.mkdir(options.outputDir)
	except OSError as e:
		#we want to handle if a directory exists. every other exception at this point is thrown again.
		if e.errno != errno.EEXIST:
			raise e
		pass

	#   get the callgraph information
	cgDoc = parseString(getDataFromFile(options.callgraph), immutable=False)

	#   build up implementationNamesByTemplateName
	implementationNamesByTemplateName = None
	try:
		implementationNamesByTemplateName = json.loads(getDataFromFile(options.implementation))
	except ValueError as e:
		logging.critical('Error decoding implementation json (%s): %s' \
			%(str(options.implementation), str(e))
		)
		sys.exit(1)
	except Exception as e:
		logging.critical('Could not interpret implementation parameter as json file to read. Trying to use it as an implementation name directly')
		implementationNamesByTemplateName = {'default':options.implementation}
	logging.debug('Initializing H90toF90Converter with the following implementations: %s' %(json.dumps(implementationNamesByTemplateName)))
	implementationsByTemplateName = dict(
		(templateName, getattr(implementations.fortran, implementationNamesByTemplateName[templateName])(optionFlags))
		for templateName in implementationNamesByTemplateName.keys()
	)

	#   parse the @domainDependant symbol declarations flags in all h90 files
	#   -> update the callgraph document with this information.
	#   note: We do this, since for simplicity reasons, the declaration parser relies on the symbol names that
	#   have been declared in @domainDependant directives. Since these directives come *after* the declaration,
	#   we need this pass
	# cgDoc = getClonedDocument(cgDoc)
	for fileNum, fileInDir in enumerate(filesInDir):
		parser = H90XMLSymbolDeclarationExtractor(cgDoc, implementationsByTemplateName=implementationsByTemplateName)
		parser.processFile(fileInDir)
		logging.debug("Symbol declarations extracted for " + fileInDir + "")
		printProgressIndicator(sys.stderr, fileInDir, fileNum + 1, len(filesInDir), "Symbol parsing, excluding imports")
	progressIndicatorReset(sys.stderr)

	#   build up symbol table indexed by module name
	moduleNodesByNameWithoutImplicitImports = getModuleNodesByName(cgDoc)
	symbolAnalyzer = SymbolDependencyAnalyzer(cgDoc)
	symbolAnalysisByRoutineNameAndSymbolNameWithoutImplicitImports = symbolAnalyzer.getSymbolAnalysisByRoutine()
	symbolsByModuleNameAndSymbolNameWithoutImplicitImports = getSymbolsByModuleNameAndSymbolName(
		ImmutableDOMDocument(cgDoc),
		moduleNodesByNameWithoutImplicitImports,
		symbolAnalysisByRoutineNameAndSymbolName=symbolAnalysisByRoutineNameAndSymbolNameWithoutImplicitImports
	)

	#   parse the symbols again, this time know about all informations in the sourced modules in import
	#   -> update the callgraph document with this information.
	for fileNum, fileInDir in enumerate(filesInDir):
		parser = H90XMLSymbolDeclarationExtractor(
			cgDoc,
			symbolsByModuleNameAndSymbolNameWithoutImplicitImports,
			implementationsByTemplateName=implementationsByTemplateName
		)
		parser.processFile(fileInDir)
		logging.debug("Symbol imports and declarations extracted for " + fileInDir + "")
		printProgressIndicator(sys.stderr, fileInDir, fileNum + 1, len(filesInDir), "Symbol parsing, including imports")
	progressIndicatorReset(sys.stderr)

	fileContents = makeFileContent(cgDoc, filesInDir, implementationsByTemplateName)
	modulesByName, routinesByName = makeCallGraphAnalysis(fileContents)
	prepareImplementation(fileContents)
	convertEverything(fileContents, modulesByName, routinesByName)

	# from timeit import timeit
	# try:
	# 	sys.stderr.write("Starting final conversion process\n")
	# 	sys.stderr.write("Conversion process finished after: {:10.1f} seconds\n".format(timeit(convertEverything, number=1)))
	# except UsageError as e:
	# 	logging.error('Error: %s' %(str(e)))
	# 	sys.exit(1)

##################### MAIN ##############################
#get all program arguments
parser = OptionParser()
parser.add_option("-i", "--sourceDir", dest="sourceDir",
									help="Source directory containing all h90 files for this implementation")
parser.add_option("-o", "--outputDir", dest="outputDir",
									help="Output directory to store all the P90 files generated by this script")
parser.add_option("-c", "--callgraph", dest="callgraph",
									help="analyzed callgraph XML file to read", metavar="XML")
parser.add_option("-d", "--debug", action="store_true", dest="debug",
									help="show debug print in standard error output")
parser.add_option("-m", "--implementation", dest="implementation",
									help="specify either a FortranImplementation classname or a JSON containing classnames by template name and a 'default' entry", metavar="IMP")
parser.add_option("--optionFlags", dest="optionFlags",
									help="can be used to switch on or off the following flags (comma separated): DO_NOT_TOUCH_GPU_CACHE_SETTINGS")
(options, args) = parser.parse_args()

setupDeferredLogging('preprocessor.log', logging.DEBUG if options.debug else logging.INFO, showDeferredLogging=not options.debug)

optionFlags = [flag for flag in options.optionFlags.split(',') if flag not in ['', None]] if options.optionFlags != None else []
logging.debug('Option Flags: %s' %(optionFlags))
if options.debug and 'DEBUG_PRINT' not in optionFlags:
	optionFlags.append('DEBUG_PRINT')

if (not options.sourceDir):
		logging.error("sourceDir option is mandatory. Use '--help' for informations on how to use this module")
		sys.exit(1)

if (not options.outputDir):
		logging.error("outputDir option is mandatory. Use '--help' for informations on how to use this module")
		sys.exit(1)

if (not options.callgraph):
		logging.error("callgraph option is mandatory. Use '--help' for informations on how to use this module")
		sys.exit(1)

if (not options.implementation):
	logging.error("implementation option is mandatory. Use '--help' for informations on how to use this module")
	sys.exit(1)

generateCodbase(options)

# codeSanitizer = FortranCodeSanitizer()
# for fileNum, fc in enumerate(fileContents):
# 	outputPath = os.path.join(
# 		os.path.normpath(options.outputDir),
# 		os.path.splitext(os.path.basename(fc['fileName']))[0] + ".P90.temp"
# 	)
# 	printProgressIndicator(sys.stderr, fc['fileName'], fileNum + 1, len(fileContents), "Implementing as Standard Fortran")
# 	outputStream = FileIO(outputPath, mode="wb")
# 	try:
# 		outputStream.write(codeSanitizer.sanitizeLines(fc['prefix'] + "\n"))
# 		for m in fc['modules']:
# 			outputStream.write(codeSanitizer.sanitizeLines(m.implemented(modulesByName, routinesByName) + "\n\n"))
# 			outputStream.write(codeSanitizer.sanitizeLines(fc['appendixByModuleName'].get(m.name, "") + "\n"))
# 	except UsageError as e:
# 		logging.error('Error: %s' %(str(e)))
# 		sys.exit(1)
# 	finally:
# 		outputStream.close()
# progressIndicatorReset(sys.stderr)