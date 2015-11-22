import re
import sys
from optparse import OptionParser
from xml.dom.minidom import Document
from DomHelper import getDomainDependantTemplatesAndEntries, parseString

def isEqualElement(a, b, ignoreAttributes):
  if a.tagName!=b.tagName:
      return False

  if len(a.attributes.keys()) != len(b.attributes.keys()):
    return False
  for aatKey, batKey in zip(sorted(a.attributes.keys()), sorted(b.attributes.keys())):
    if aatKey in ignoreAttributes and batKey == aatKey:
      continue
    if aatKey != batKey:
      if options.debug:
        sys.stderr.write("equality fails for node attributes %s compared to %s; key: %s vs. %s\n" \
          %(a.toxml(), b.toxml(), aatKey, batKey))
      return False
    if (a.attributes.get(aatKey) == None and b.attributes.get(aatKey) != None) \
    or (a.attributes.get(aatKey) != None and b.attributes.get(aatKey) == None):
      if options.debug:
        sys.stderr.write("equality fails for node attributes %s compared to %s; one is None, the other not\n" \
            %(a.toxml(), b.toxml()))
      return False
    if a.attributes.get(aatKey).value != b.attributes.get(aatKey).value:
      if options.debug:
          sys.stderr.write("equality fails for node attributes %s compared to %s; value: %s vs. %s" \
            %(a.toxml(), b.toxml(), a.attributes.get(aatKey).value, b.attributes.get(aatKey).value))
      return False

  if len(a.childNodes)!=len(b.childNodes):
    return False

  for ac, bc in zip(a.childNodes, b.childNodes):
    if ac.nodeType != bc.nodeType:
      return False
    if ac.nodeType == ac.TEXT_NODE and ac.data != bc.data:
      return False
    if ac.nodeType == ac.ELEMENT_NODE and not isEqualElement(ac, bc, ignoreAttributes):
      return False

  return True

def getRoutinesBySource(xmlData):
	result = {}
	for routine in xmlData.getElementsByTagName('routine'):
		sourceName = routine.getAttribute('source')
		if sourceName in ['', None]:
			raise Exception("invalid source name definition in routine node %s" %(routine.toxml()))
		if result.get(sourceName) == None:
			result[sourceName] = [routine]
		else:
			result[sourceName].append(routine)
	return result

def getRoutineBySourceAndName(xmlData):
	result = {}
	for routine in xmlData.getElementsByTagName('routine'):
		sourceName = routine.getAttribute('source')
		routineName = routine.getAttribute('name')
		if sourceName in ['', None] or routineName in ['', None]:
			raise Exception("invalid routine node %s" %(routine.toxml()))
		if result.get((sourceName, routineName)) == None:
			result[(sourceName, routineName)] = routine
		else:
			raise Exception("routine name %s is used twice in source %s - Hybrid Fortran needs distinct routine names per source." %(
				routineName, sourceName
			))
	return result

def getSourcesWithParallelRegionPositionChanges(inputXML, referenceXML):
	def isSourceStillValid(source, referenceRoutinesBySource, inputRoutinesBySourceAndName):
		referenceRoutines = referenceRoutinesBySource[source]
		for referenceRoutine in referenceRoutines:
			routineName = referenceRoutine.getAttribute('name')
			if routineName in [None, '']:
				raise Exception("invalid name definition in reference routine node %s" %(referenceRoutine.toxml()))
			inputRoutine = inputRoutinesBySourceAndName.get((source, routineName))
			if inputRoutine == None:
				if options.debug:
					sys.stderr.write("routine %s has been deleted from %s\n" %(routineName, source))
				return False
			if inputRoutine.getAttribute('parallelRegionPosition') != referenceRoutine.getAttribute('parallelRegionPosition'):
				if options.debug:
					sys.stderr.write("routine %s in source %s does not have the same parallel region position as before\n" %(
						routineName, source
					))
				return False
		return True

	referenceRoutinesBySource = getRoutinesBySource(referenceXML)
	inputRoutinesBySourceAndName = getRoutineBySourceAndName(inputXML)
	return [
		source for source in referenceRoutinesBySource.keys()
		if not isSourceStillValid(source, referenceRoutinesBySource, inputRoutinesBySourceAndName)
	]

def getDomainDependantTemplatesAndEntriesByModuleNameAndSymbolName(xmlData):
	result = {}
	for module in xmlData.getElementsByTagName('module'):
		moduleName = module.getAttribute('name')
		templatesAndEntries = getDomainDependantTemplatesAndEntries(xmlData, module)
		for template, entry in templatesAndEntries:
			symbolName = entry.firstChild.nodeValue
			result[(moduleName,symbolName)] = (template, entry)
	return result

def getModulesByName(xmlData):
	result = {}
	for module in xmlData.getElementsByTagName('module'):
		moduleName = module.getAttribute('name')
		result[moduleName] = module
	return result

def getSourcesWithModuleSymbolChanges(inputXML, referenceXML):
	def isModuleStillValid(module,inputModules,inputModuleSymbolIndex,referenceXML):
		moduleName = module.getAttribute('name')
		if moduleName in ["", None]:
			raise Exception("invalid module definition: %s" %(module.toxml()))
		if not moduleName in inputModules:
			return True #module is not found anymore - we don't have to care about this case, should be handled by GNU Make
		templatesAndEntries = getDomainDependantTemplatesAndEntries(referenceXML, module)
		for template, entry in templatesAndEntries:
			symbolName = entry.firstChild.nodeValue
			if symbolName in ["", None]:
				raise Exception("invalid entry: %s" %(entry.toxml()))
			inputTemplateAndEntry = inputModuleSymbolIndex.get((moduleName, symbolName))
			if inputTemplateAndEntry == None:
				if options.debug:
					sys.stderr.write("symbol %s has been deleted from %s\n" %(
						symbolName, moduleName
					))
				return False #symbol has been deleted
			inputTemplate,inputEntry = inputModuleSymbolIndex[(moduleName, symbolName)]
			if not isEqualElement(entry, inputEntry, ["id"]) or not isEqualElement(template, inputTemplate, ["id"]):
				if options.debug:
					sys.stderr.write("symbol %s in %s has been changed\n" %(
						symbolName, moduleName
					))
				return False
		return True

	inputModuleSymbolIndex = getDomainDependantTemplatesAndEntriesByModuleNameAndSymbolName(inputXML)
	inputModules = getModulesByName(inputXML)
	return [
		module.getAttribute('name')
		for module in referenceXML.getElementsByTagName('module')
		if not isModuleStillValid(module,inputModules,inputModuleSymbolIndex,referenceXML)
	]

def getSourcesByUsedSource(xmlData):
	result = {}
	for entry in referenceXML.getElementsByTagName('entry'):
		sourceName = entry.parentNode.parentNode.parentNode.getAttribute('source')
		if sourceName in [None, '']:
			continue
		usedSource = entry.getAttribute('sourceModule')
		if usedSource in result:
			result[usedSource].append(sourceName)
		else:
			result[usedSource] = [sourceName]
	return result

def getSourcesToUpdateForModuleSymbolChanges(inputXML, referenceXML):
	sourcesWithModuleSymbolChanges = getSourcesWithModuleSymbolChanges(inputXML, referenceXML)
	sourcesByUsedSource = getSourcesByUsedSource(inputXML)
	sourcesToUpdateKeyed = {}
	for source in sourcesWithModuleSymbolChanges:
		sourcesUsingIt = sourcesByUsedSource.get(source)
		if sourcesUsingIt == None:
			continue
		for sourceUsingIt in sourcesUsingIt:
			sourcesToUpdateKeyed[sourceUsingIt] = None
	return sourcesToUpdateKeyed.keys()

##################### MAIN ##############################
#get all program arguments
parser = OptionParser()
parser.add_option("-r", "--reference", dest="reference", help="reference callgraph")
parser.add_option("-i", "--input", dest="input", help="input callgraph to be analysed")
parser.add_option("-d", "--debug", action="store_true", dest="debug", help="show debug print in standard error output")
(options, args) = parser.parse_args()

if (not options.reference or not options.input):
	raise Exception("Missing options. Please use '-h' option to see usage.")

inputXML = None
referenceXML = None

inputXMLFile = open(str(options.input),'r')
inputXMLData = inputXMLFile.read()
inputXMLFile.close()
inputXML = parseString(inputXMLData)
referenceXMLFile = None
try:
	referenceXMLFile = open(str(options.reference),'r')
except Exception:
	pass
try:
	if referenceXMLFile != None:
		referenceXMLData = referenceXMLFile.read()
		referenceXMLFile.close()
		referenceXML = parseString(referenceXMLData)
		sourcesToUpdateKeyed = {}
		for source in getSourcesWithParallelRegionPositionChanges(inputXML, referenceXML):
			sourcesToUpdateKeyed[source] = None
		for source in getSourcesToUpdateForModuleSymbolChanges(inputXML, referenceXML):
			sourcesToUpdateKeyed[source] = None
		print(
			" ".join(sourcesToUpdateKeyed.keys())
		)
	else:
		print(
			" ".join(getRoutinesBySource(inputXML).keys())
		)
except Exception, e:
  sys.stderr.write('Error when generating analysing, which sources are to be reprocessed: %s\n' %(str(e)))
  sys.exit(1)



