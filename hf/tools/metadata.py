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

from xml.dom.minidom import Document, Node, parseString as parseStringUsingMinidom
from tools.commons import BracketAnalyzer, enum
import uuid
import re
import logging

domainDependantAttributes = ["autoDom", "present", "transferHere"]

class ParallelRegionDomain(object):
    def __init__(self, name, size, startsAt=None, endsAt=None):
        self.name = name
        self.size = size
        self.startsAt = startsAt
        self.endsAt = endsAt

class DOMNode(Node):
    pass

class DOMDocument(Document):
    pass

class ImmutableDOMNode(object):
    def __init__(self, node):
        self.node = node
        self.cachedElementsByTagName = {}

    @property
    def nodeType(self):
        return self.node.nodeType

    @property
    def nodeName(self):
        return self.node.nodeName

    @property
    def nodeValue(self):
        return self.node.nodeValue

    @property
    def attributes(self):
        return self.node.attributes

    @property
    def parentNode(self):
        return ImmutableDOMNode(self.node.parentNode)

    @property
    def firstChild(self):
        return ImmutableDOMNode(self.node.firstChild)

    @property
    def childNodes(self):
        return [ImmutableDOMNode(node) for node in self.node.childNodes]

    def getElementsByTagName(self, name):
        result = self.cachedElementsByTagName.get(name)
        if result != None:
            return result
        uncachedResult = self.node.getElementsByTagName(name)
        cachedResult = [ImmutableDOMNode(node) for node in uncachedResult]
        self.cachedElementsByTagName[name] = cachedResult
        return cachedResult

    def getAttribute(self, qName):
        return self.node.getAttribute(qName)

    def toxml(self, encoding=None):
        return self.node.toxml(encoding)

    def toprettyxml(self, indent="", newl="", encoding=None):
        return self.node.toprettyxml(indent, newl, encoding)

    def appendChild(self, node):
        raise Exception("cannot append child to immutable node")

    def removeChild(self, node):
        raise Exception("cannot remove child from immutable node")

    def cloneNode(self, deep):
        raise Exception("clone node not implemented")

    def createElement(self, tagName):
        raise Exception("cannot create element for immutable node")

    def createTextNode(self, tagName):
        raise Exception("cannot create text node for immutable node")

    def createAttribute(self, qName):
        raise Exception("cannot create attribute for immutable node")

class ImmutableDOMDocument(ImmutableDOMNode):
    pass

def getClonedDocument(doc):
    clone = doc.cloneNode(deep=True)
    if hasattr(clone, "_firstLevelElementCache"):
        del clone._firstLevelElementCache
    if hasattr(clone, "_templateCache"):
        del clone._templateCache
    return clone

def parseString(data, immutable=False):
    doc = parseStringUsingMinidom(data)
    if immutable:
        #this saves a lot of preprocessing time by caching the results instead of using minidom's native implementation
        return ImmutableDOMDocument(doc)
    return doc

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

def getRegionPosition(routineName, routines):
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

#TODO: refactor the call lookups in the parser to use this index
def getCallersByCalleeName(callNodes):
    callersByCalleeName = {}
    for callNode in callNodes:
        calleeName = callNode.getAttribute("callee")
        currCallerList = callersByCalleeName.get(calleeName)
        if currCallerList == None:
            currCallerList = [callNode]
            callersByCalleeName[calleeName] = currCallerList
        else:
            currCallerList.append(callNode)
    return callersByCalleeName

def getCalleesByCallerName(callNodes):
    calleesByRoutineName = {}
    for callNode in callNodes:
        callerName = callNode.getAttribute("caller")
        currCalleeList = calleesByRoutineName.get(callerName)
        if currCalleeList == None:
            currCalleeList = [callNode]
            calleesByRoutineName[callerName] = currCalleeList
        else:
            currCalleeList.append(callNode)
    return calleesByRoutineName

def createOrGetFirstNodeWithName(nodeName, doc):
    nodeArr = doc.getElementsByTagName(nodeName)
    node = None
    if (len(nodeArr) > 0):
        node = nodeArr[0]
    else:
        node = doc.createElement(nodeName)
        doc.firstChild.appendChild(node)
    return node

def getNodeValue(node, mergeEntries=False):
    if hasattr(node, "nodeValue") and node.nodeValue != None:
        return node.nodeValue
    elif node.childNodes and len(node.childNodes) == 1 and hasattr(node.childNodes[0], "nodeValue"):
        return node.childNodes[0].nodeValue
    elif mergeEntries and all([n.tagName == "entry" for n in node.childNodes]):
        return ",".join([getNodeValue(n).strip() for n in node.childNodes])
    return None

def appendSeparatedTextAsNodes(text, separator, doc, parent, nodeName):
    if text == None:
        raise Exception("cannot append None as entry")
    payload = text.strip()
    if payload == '':
        raise Exception("cannot append empty node")
    payloadList = payload.split(separator)
    appendedNodes = []
    for entry in payloadList:
        stripped = entry.strip()
        if stripped == "":
            continue
        entryNode = doc.createElement(nodeName)
        textNode = doc.createTextNode(stripped)
        entryNode.appendChild(textNode)
        parent.appendChild(entryNode)
        appendedNodes.append(entryNode)
    return appendedNodes

def addAndGetEntries(doc, parent, commaSeparatedEntries):
    return appendSeparatedTextAsNodes(commaSeparatedEntries, ",", doc, parent, "entry")

def setDomainDependants(doc, parent, specificationText, entryText):
    relationNode, template = setTemplateInfos(
        doc,
        parent=parent,
        specText=specificationText,
        templateParentNodeName="domainDependantTemplates",
        templateNodeName="domainDependantTemplate",
        referenceParentNodeName="domainDependants"
    )
    entries = addAndGetEntries(doc, relationNode, entryText)
    return relationNode, template, entries


def deduplicateRelations(parent):
    pass

def hasDuplicateAttributes(node1, node2, ignoreIDs=True):
    if hasattr(node1, "tagName") and ((not hasattr(node2, "tagName")) or node1.tagName != node2.tagName):
        return False
    if not node2.attributes or len(node2.attributes) == 0:
        if not node1.attributes or len(node1.attributes) == 0:
            #both nodes don't have any attributes
            return True
        else:
            return False
    if len(node2.attributes) != len(node1.attributes):
        return False
    for (name, value) in node1.attributes.items():
        #we don't care for id attributes. That is, if two otherwise identical nodes have different ids,
        #they shall not be called unique.
        if ignoreIDs and name == "id":
            continue
        newAttributeValue = node2.getAttribute(name)
        if (value != newAttributeValue):
            break
    else:
        #the inner loop has never breaked -> both nodes share all attributes.
        return True
    return False

def firstDuplicateChild(parent, newNode, cgDoc=None, ignoreIDs=True):
    '''Get first duplicate for the newNode within parent's childNodes'''
    ## trying to make this faster, but with th erecursiveness it's tricky
    # currAttributes = OrderedDict()
    # if hasattr(newNode, "tagName"):
    #     currAttributes["tagName"] = newNode.tagName
    # currAttributeNames = sorted([name for (name, value) in newNode.attributes.items() if name != "id"])
    # for attributeName in currAttributeNames:
    #     currAttributes[attributeName] = newNode.attributes[attributeName]
    # attributeTuple = tuple(currAttributes.values())
    # attributeCache = None
    # if cgDoc and hasattr(cgDoc, "_attributeCache")
    #     attributeCache = cgDoc._attributeCache
    # elif cgDoc:
    #     attributeCache = {}
    #     cgDoc._attributeCache = attributeCache
    # if attributeCache and attributeTuple in attributeCache:
    #     return attributeCache[attributeTuple]
    # if attributeCache:
    #     attributeCache[attributeTuple] = newNode
    #     return None

    nodesWithDuplicateAttributes = []
    for childNode in parent.childNodes:
        if hasDuplicateAttributes(childNode, newNode, ignoreIDs):
            nodesWithDuplicateAttributes.append(childNode)
    if len(nodesWithDuplicateAttributes) == 0:
        return None

    newNodeContent = {}
    newNodeValue = getNodeValue(newNode, mergeEntries=True)
    if newNodeValue:
        newNodeContent["value"] = newNodeValue

    #Let's check the children of all nodes with duplicate attributes
    for node in nodesWithDuplicateAttributes:
        if len(newNode.childNodes) != len(node.childNodes):
            continue
        newNodeContentCopy = newNodeContent.copy()
        nodeValue = getNodeValue(node, mergeEntries=True)
        if newNodeContentCopy.get("value", None) != None:
            if not nodeValue:
                continue
            elif newNodeContentCopy.get("value", None) == nodeValue:
                newNodeContentCopy["value"] = "match!"

        #for each of the children of newNode, check whether they are unique among the current nodes' children.
        for newChildNode in newNode.childNodes:
            if firstDuplicateChild(node, newChildNode) != None:
                if hasattr(newChildNode, "tagName"):
                    newNodeContentCopy[newChildNode.tagName + str(uuid.uuid4())] = "match!"
                else:
                    newNodeContentCopy["value"] = "match!"
            else:
                if hasattr(newChildNode, "tagName"):
                    newNodeContentCopy[newChildNode.tagName + str(uuid.uuid4())] = "nomatch"
                else:
                    newNodeContentCopy["value"] = "nomatch"

        newNodeContentTags = newNodeContentCopy.keys()
        for newNodeContentTag in newNodeContentTags:
            if newNodeContentCopy[newNodeContentTag] != "match!":
                break
        else:
            return node
    return None

def getAttributesDomainsDeclarationPrefixAndMacroNames(moduleTemplate, procedureTemplate):
    attributesModule = getAttributes(moduleTemplate)
    attributesProcedure = getAttributes(procedureTemplate)
    finalAttributes = []
    domainsFromTemplate = []
    if "autoDom" in attributesModule and "autoDom" in attributesProcedure:
        finalAttributes.append("autoDom")
        domainsFromTemplate = getDomNameAndSize(procedureTemplate)
    elif not "autoDom" in attributesProcedure:
        domainsFromTemplate = getDomNameAndSize(procedureTemplate)
    elif not "autoDom" in attributesModule:
        domainsFromTemplate = getDomNameAndSize(moduleTemplate)
    for attribute in domainDependantAttributes:
        if attribute == "autoDom":
            continue
        if attribute in attributesProcedure:
            finalAttributes.append(attribute)

    moduleDeclarationPrefix = getStringProperty(moduleTemplate, 'declarationPrefix')
    procedureDeclarationPrefix = getStringProperty(procedureTemplate, 'declarationPrefix')
    finalDeclarationPrefix = moduleDeclarationPrefix if procedureDeclarationPrefix == None else procedureDeclarationPrefix

    moduleAccPP = getStringProperty(moduleTemplate, 'accPP')
    procedureAccPP = getStringProperty(procedureTemplate, 'accPP')
    finalAccPP = moduleAccPP if procedureAccPP == None else procedureAccPP

    moduleDomPP = getStringProperty(moduleTemplate, 'domPP')
    procedureDomPP = getStringProperty(procedureTemplate, 'domPP')
    finalDomPP = moduleDomPP if procedureDomPP == None else procedureDomPP

    return finalAttributes, domainsFromTemplate, finalDeclarationPrefix, finalAccPP, finalDomPP

def getOrCreateFirstLevelElement(doc, nodeName):
    if not hasattr(doc, '_firstLevelElementCache'):
        doc._firstLevelElementCache = {}
    element = doc._firstLevelElementCache.get(nodeName)
    if element:
        return element
    elements = doc.getElementsByTagName(nodeName)
    if len(elements) > 0:
        element = elements[0]
        doc._firstLevelElementCache[nodeName] = element
        return element
    element = doc.createElement(nodeName)
    doc.documentElement.appendChild(element)
    doc._firstLevelElementCache[nodeName] = element
    return element

def updateAndGetParallelRegionInfo(doc, subroutineNode, parallelRegionSpecification, startLine):
    relationNode, templateNode = setTemplateInfos(
        doc,
        subroutineNode,
        parallelRegionSpecification,
        "parallelRegionTemplates",
        "parallelRegionTemplate",
        "parallelRegions"
    )
    relationNode.setAttribute("startLine", str(startLine))
    return relationNode, templateNode

def setTemplateInfos(doc, parent, specText, templateParentNodeName, templateNodeName, referenceParentNodeName):
    if not parent:
        raise Exception("cannot add template info '%s' to nonexisting parent" %(specText))

    templateLibrary = getOrCreateFirstLevelElement(doc, templateParentNodeName)
    templateNode = doc.createElement(templateNodeName)
    templateNode.setAttribute("id", str(uuid.uuid4()))

    settingPattern = re.compile(r'[\s,]*(\w*)\s*(\(.*)')
    remainder = specText
    while len(remainder) > 0:
        settingMatch = settingPattern.match(remainder)
        if not settingMatch:
            break
        name = settingMatch.group(1).strip()
        settingNode = doc.createElement(name)
        templateNode.appendChild(settingNode)
        textAfterSettingName = settingMatch.group(2)
        settingBracketAnalyzer = BracketAnalyzer()
        settingText, remainder = settingBracketAnalyzer.getTextWithinBracketsAndRemainder(textAfterSettingName)
        addAndGetEntries(doc, settingNode, settingText)

    #deduplicate the definition
    duplicateTemplateNode = firstDuplicateChild(templateLibrary, templateNode)
    if duplicateTemplateNode:
        templateNode = duplicateTemplateNode
    else:
        templateLibrary.appendChild(templateNode)
        if not hasattr(doc, "_templateCache"):
            doc._templateCache = {}
        templatesByID = doc._templateCache.get(templateNodeName, {})
        templatesByID[templateNode.getAttribute("id")] = templateNode
        doc._templateCache[templateNodeName] = templatesByID

    templateID = templateNode.getAttribute("id")
    referenceParentNodes = parent.getElementsByTagName(referenceParentNodeName)
    referenceParentNode = None
    if len(referenceParentNodes) == 0:
        referenceParentNode = doc.createElement(referenceParentNodeName)
        parent.appendChild(referenceParentNode)
    else:
        referenceParentNode = referenceParentNodes[0]
    relationNode = doc.createElement("templateRelation")
    relationNode.setAttribute("id", templateID)
    referenceParentNode.appendChild(relationNode)
    return relationNode, templateNode

def regionTemplatesByID(cgDoc, templateTypeName):
    regionTemplatesByID = None
    if hasattr(cgDoc, "_templateCache"):
        regionTemplatesByID = cgDoc._templateCache.get(templateTypeName)
    else:
        cgDoc._templateCache = {}
    if regionTemplatesByID:
        return regionTemplatesByID
    regionTemplatesByID = {}
    templateNodes = cgDoc.getElementsByTagName(templateTypeName)
    for templateNode in templateNodes:
        idStr = templateNode.getAttribute('id')
        if not idStr or idStr == '':
            raise Exception("Template definition without id attribute.")
        regionTemplatesByID[idStr] = templateNode
    cgDoc._templateCache[templateTypeName] = regionTemplatesByID
    return regionTemplatesByID

RoutineNodeInitStage = enum("NO_DIRECTIVES",
    "DIRECTIVES_WITH_PARALLELREGION_POSITION",
    "DIRECTIVES_WITHOUT_PARALLELREGION_POSITION",
    "UNDEFINED"
)

def getRoutineNodeInitStage(routineNode):
    hasActiveParallelRegions = False
    hasDomainDependantDirectives = False

    parallelRegionsParents = routineNode.getElementsByTagName('activeParallelRegions')
    if parallelRegionsParents and len(parallelRegionsParents) > 0 \
    and len(parallelRegionsParents[0].getElementsByTagName("templateRelation")) > 0:
        hasActiveParallelRegions = True
    domainDependantRelationsParents = routineNode.getElementsByTagName("domainDependants")
    if domainDependantRelationsParents and len(domainDependantRelationsParents) > 0 \
    and len(domainDependantRelationsParents[0].getElementsByTagName("templateRelation")) > 0:
        hasDomainDependantDirectives = True

    if not hasActiveParallelRegions and not hasDomainDependantDirectives:
        return RoutineNodeInitStage.NO_DIRECTIVES

    parallelRegionPostion = routineNode.getAttribute("parallelRegionPosition")
    if not parallelRegionPostion or parallelRegionPostion.strip() == "":
        return RoutineNodeInitStage.DIRECTIVES_WITHOUT_PARALLELREGION_POSITION

    return RoutineNodeInitStage.DIRECTIVES_WITH_PARALLELREGION_POSITION

def getDomNameAndSize(templateNode):
    dimensionSizesTemplateNodes = templateNode.getElementsByTagName("domSize")
    if not dimensionSizesTemplateNodes or len(dimensionSizesTemplateNodes) == 0:
        return []
    dimensionSizesInTemplate = [node.firstChild.nodeValue for node in dimensionSizesTemplateNodes[0].getElementsByTagName("entry")]
    dimensionNamesTemplateNodes = templateNode.getElementsByTagName("domName")
    if not dimensionNamesTemplateNodes or len(dimensionNamesTemplateNodes) == 0:
        return []
    dimensionNamesInTemplate = [node.firstChild.nodeValue for node in dimensionNamesTemplateNodes[0].getElementsByTagName("entry")]
    if len(dimensionNamesInTemplate) != len(dimensionSizesInTemplate):
        raise Exception("Number of domain names does not match number of domain sizes specified; Domain names: %s, domain sizes: %s" %(dimensionNamesInTemplate, dimensionSizesInTemplate))
    #map the domNames to the sizes declared in the declaration
    dimensionNameAndSize = []
    for i in range(len(dimensionNamesInTemplate)):
        dimensionNameAndSize.append((dimensionNamesInTemplate[i], dimensionSizesInTemplate[i]))
    return dimensionNameAndSize

def getDeclarationPrefix(templateNode):
    declarationPrefixTemplateNodes = templateNode.getElementsByTagName("declarationPrefix")
    if not declarationPrefixTemplateNodes or len(declarationPrefixTemplateNodes) == 0:
        return None
    declarationPrefixesInTemplate = [node.firstChild.nodeValue for node in declarationPrefixTemplateNodes[0].getElementsByTagName("entry")]
    return declarationPrefixesInTemplate[0]

def getStringProperty(templateNode, propertyName):
    templateNodes = templateNode.getElementsByTagName(propertyName)
    if not templateNodes or len(templateNodes) == 0:
        return None
    propertyValues = [node.firstChild.nodeValue for node in templateNodes[0].getElementsByTagName("entry")]
    return propertyValues[0]

def getAttributes(templateNode):
    attributesTemplateNodes = templateNode.getElementsByTagName("attribute")
    if not attributesTemplateNodes or len(attributesTemplateNodes) == 0:
        return []
    return [node.firstChild.nodeValue for node in attributesTemplateNodes[0].getElementsByTagName("entry")]

def getDomainDependantTemplatesAndEntries(cgDoc, routineNode):
    result = []
    domainDependantTemplateByID = regionTemplatesByID(cgDoc, "domainDependantTemplate")
    domainDependantRelationsParent = routineNode.getElementsByTagName("domainDependants")
    if not domainDependantRelationsParent or len(domainDependantRelationsParent) == 0:
        return result
    domainDependantRelations = domainDependantRelationsParent[0].getElementsByTagName("templateRelation")
    for domainDependantRelation in domainDependantRelations:
        entries = domainDependantRelation.getElementsByTagName("entry")
        templateID = domainDependantRelation.getAttribute("id")
        template = domainDependantTemplateByID.get(templateID)
        if not template:
            raise Exception("domain dependant template with ID %s not found, referenced by symbols %s in routine %s" %(
                templateID,
                [entry.firstChild.nodeValue for entry in entries],
                routineNode.getAttribute("name")
            ))
        for entry in entries:
            if not entry.firstChild:
                raise Exception("DependantName undefined in subroutine %s, Entry: %s" \
                    %(self.currSubprocName, entry.toxml()))
            result.append((template, entry))
    return result

def getDomainsWithParallelRegionTemplate(parallelRegionTemplate):
    def getAttributeEntries(attributeName, mandatory=False, expectedLength=None):
        domNodes = parallelRegionTemplate.getElementsByTagName(attributeName)
        if mandatory and (domNodes == None or len(domNodes) != 1):
            raise Exception('%s definition is invalid in parallelRegion\nTemplate:%s' %(attributeName, parallelRegionTemplate.toprettyxml()))
        elif not mandatory and (domNodes == None or len(domNodes) == 0):
            return None
        nodeEntries = domNodes[0].getElementsByTagName('entry')
        if expectedLength != None and len(nodeEntries) != expectedLength:
            raise Exception('invalid %s definition: has %i entries - %i are expected' %(attributeName, len(nodeEntries), expectedLength))
        result = []
        for entry in nodeEntries:
            result.append(entry.firstChild.nodeValue)
        return result
    domainNames = getAttributeEntries('domName', mandatory=True)
    domainSizes = getAttributeEntries('domSize', mandatory=True, expectedLength=len(domainNames))
    startsAtEntries = getAttributeEntries('startAt', mandatory=False, expectedLength=len(domainNames))
    endsAtEntries = getAttributeEntries('endAt', mandatory=False, expectedLength=len(domainNames))
    domains = []
    for index, domainName in enumerate(domainNames):
        domains.append(ParallelRegionDomain(
            name=domainName,
            size=domainSizes[index],
            startsAt=startsAtEntries[index] if startsAtEntries != None else None,
            endsAt=endsAtEntries[index] if endsAtEntries != None else None
        ))
    return domains

def appliesTo(appliesToTests, parallelRegionTemplate):
    appliesToNodes = parallelRegionTemplate.getElementsByTagName("appliesTo")
    if not appliesToNodes or len(appliesToNodes) == 0:
        return True
    entries = appliesToNodes[0].getElementsByTagName("entry")
    if not entries or len(entries) == 0:
        raise Exception("Unexpected parallel region template definition: AppliesTo node without entry.")

    for entry in entries:
        for appliesToTest in appliesToTests:
            if entry.firstChild.nodeValue == appliesToTest:
                return True
    return False

def getTemplate(parallelRegionTemplate):
    templateNodes = parallelRegionTemplate.getElementsByTagName("template")
    if not templateNodes or len(templateNodes) == 0:
        return ''
    if len(templateNodes) != 1:
        raise Exception("Multiple templates are not supported.")
    entries = templateNodes[0].getElementsByTagName("entry")
    if len(entries) > 1:
        raise Exception("Multiple templates are not supported.")
    if len(entries) == 0:
        raise Exception("Empty template attribute is not allowed.")
    return entries[0].firstChild.nodeValue.strip()

def getReductionScalarsByOperator(parallelRegionTemplate):
    result = {}
    reductionNodes = parallelRegionTemplate.getElementsByTagName("reduction")
    if not reductionNodes or len(reductionNodes) == 0:
        return result
    reductionSpecifications = []
    for reductionNode in reductionNodes:
        entries = reductionNode.getElementsByTagName("entry")
        if len(entries) > 1:
            raise Exception("Multiple scalars per reduction specification are not supported. Please split into multiple reduction attributes.")
        if len(entries) == 0:
            raise Exception("Empty reduction attribute is not allowed.")
        reductionSpecifications.append(entries[0].firstChild.nodeValue.strip())
    validOperators = ['MAX', 'MIN', 'IAND', 'IOR', 'IEOR', '+', '*', '-', '.AND.', '.OR.', '.EQV.', '.NEQV.']
    for reductionSpecification in reductionSpecifications:
        operatorAndScalar = reductionSpecification.split(':')
        if len(operatorAndScalar) != 2:
            raise Exception("Reduction must be of form [operator]:[scalar symbol]")
        operator = operatorAndScalar[0].upper().strip()
        if operator not in validOperators:
            raise Exception("Reduction operator must be one of %s." %(", ".join(validOperators)))
        scalar = operatorAndScalar[1].strip()
        if scalar == "":
            raise Exception("Empty scalar in reduction not allowed.")
        if operator in result:
            result[operator].append(scalar)
        else:
            result[operator] = [scalar]
    return result

def getArguments(parentNode):
    return [
        argument.getAttribute("symbolName")
        for argument in parentNode.getElementsByTagName("argument")
    ]