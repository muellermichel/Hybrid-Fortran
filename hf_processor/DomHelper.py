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
#  Procedure        Domhelper.py                                       #
#  Date             2012/07/30                                         #
#  Author           Michel Müller (AOKI Laboratory)                    #
#**********************************************************************#


from xml.dom.minidom import Document
from GeneralHelper import BracketAnalyzer, enum
import uuid
import re

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

def getNodeValue(node):
    if hasattr(node, "nodeValue") and node.nodeValue != None:
        return node.nodeValue
    elif node.childNodes and len(node.childNodes) == 1 and hasattr(node.childNodes[0], "nodeValue"):
        return node.childNodes[0].nodeValue
    return None

def appendSeparatedTextAsNodes(text, separator, doc, parent, nodeName):
    if text == None:
        return
    payload = text.strip()
    if payload == '':
        return
    payloadList = payload.split(separator)
    for entry in payloadList:
        stripped = entry.strip()
        if stripped == "":
            continue
        entryNode = doc.createElement(nodeName)
        textNode = doc.createTextNode(stripped)
        entryNode.appendChild(textNode)
        parent.appendChild(entryNode)

def firstDuplicateChild(parent, newNode):
    '''Get first duplicate for the newNode within parent's childNodes'''

    nodesWithDuplicateAttributes = []
    for childNode in parent.childNodes:
        if hasattr(childNode, "tagName") and ((not hasattr(newNode, "tagName")) or childNode.tagName != newNode.tagName):
                continue
        if (not newNode.attributes or len(newNode.attributes) == 0):
            if (not childNode.attributes or len(childNode.attributes) == 0):
                #both the new node and the current childnode don't have any attributes
                nodesWithDuplicateAttributes.append(childNode)
                continue
            else:
                #child has attributes while newNode doesn't have -> next!
                continue
        for (name, value) in childNode.attributes.items():
            #we don't care for id attributes. That is, if two otherwise identical nodes have different ids,
            #they shall not be called unique.
            if name == "id":
                continue
            newAttributeValue = newNode.getAttribute(name)
            if (value != newAttributeValue):
                break;
        else:
            #the inner loop has never breaked -> current child shares all attribute values with newNode.
            nodesWithDuplicateAttributes.append(childNode)
            break

    if len(nodesWithDuplicateAttributes) == 0:
        return None

    newNodeContent = {}
    newNodeValue = getNodeValue(newNode)
    if newNodeValue:
        newNodeContent["value"] = newNodeValue

    #Let's check the children of all nodes with duplicate attributes
    for node in nodesWithDuplicateAttributes:
        if len(newNode.childNodes) != len(node.childNodes):
            continue

        newNodeContentCopy = newNodeContent.copy()
        nodeValue = getNodeValue(node)

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

def setTemplateInfos(doc, parent, specText, templateParentNodeName, templateNodeName, referenceParentNodeName):
    templateLibraries = doc.getElementsByTagName(templateParentNodeName)
    templateLibrary = None
    if len(templateLibraries) == 0:
        templateLibrary = doc.createElement(templateParentNodeName)
        doc.documentElement.appendChild(templateLibrary)
    else:
        templateLibrary = templateLibraries[0]

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
        settingText, remainder = settingBracketAnalyzer.splitAfterClosingBrackets(textAfterSettingName)
        #cut away the left and right bracket
        settingText = settingText.partition("(")[2]
        settingText = settingText.rpartition(")")[0]
        appendSeparatedTextAsNodes(settingText, ',', doc, settingNode, 'entry')

    #deduplicate the parallel region definition
    duplicateTemplateNode = firstDuplicateChild(templateLibrary, templateNode)
    if duplicateTemplateNode:
        templateNode = duplicateTemplateNode
    else:
        templateLibrary.appendChild(templateNode)

    templateID = templateNode.getAttribute("id")
    referenceParentNodes = parent.getElementsByTagName(referenceParentNodeName)
    referenceParentNode = None
    if len(referenceParentNodes) == 0:
        referenceParentNode = doc.createElement(referenceParentNodeName)
        parent.appendChild(referenceParentNode)
    else:
        referenceParentNode = referenceParentNodes[0]
    entry = doc.createElement("templateRelation")
    entry.setAttribute("id", templateID)
    referenceParentNode.appendChild(entry)
    return entry

def regionTemplatesByID(cgDoc, templateTypeName):
    templateNodes = cgDoc.getElementsByTagName(templateTypeName)
    regionTemplatesByID = {}
    for templateNode in templateNodes:
        idStr = templateNode.getAttribute('id')
        if not idStr or idStr == '':
            raise Exception("Template definition without id attribute.")
        regionTemplatesByID[idStr] = templateNode

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
        template = domainDependantTemplateByID[templateID]
        for entry in entries:
            if not entry.firstChild:
                raise Exception("Unexpected Error: DependantName undefined in subroutine %s, Entry: %s" \
                    %(self.currSubprocName, entry.toxml()))
            result.append((template, entry))
    return result

def domainNamesAndSizesWithParallelRegionTemplate(parallelRegionTemplate):
    domNodes = parallelRegionTemplate.getElementsByTagName('domName')
    if not domNodes or len(domNodes) != 1:
        raise Exception('parallel region template without valid domain definition.\nTemplate:%s' %(parallelRegionTemplate.toprettyxml()))
    entries = domNodes[0].getElementsByTagName('entry')
    domainNames = []
    for entry in entries:
        domainNames.append(entry.firstChild.nodeValue)
    domSizeNodes = parallelRegionTemplate.getElementsByTagName('domSize')
    if not domSizeNodes or len(domSizeNodes) != 1:
        raise Exception('parallel region template without valid domain size definition')
    entries = domSizeNodes[0].getElementsByTagName('entry')
    domainSizes = []
    for entry in entries:
        domainSizes.append(entry.firstChild.nodeValue)
    if len(domainSizes) != len(domainNames):
        raise Exception('number of domain size definition does not match number of domains')
    return domainNames, domainSizes

def appliesTo(appliesToTests, parallelRegionTemplate):
    appliesToNodes = parallelRegionTemplate.getElementsByTagName("appliesTo")
    if not appliesToNodes or len(appliesToNodes) == 0:
        return False
    entries = appliesToNodes[0].getElementsByTagName("entry")
    if not entries or len(entries) == 0:
        raise Exception("Unexpected parallel region template definition: AppliesTo node without entry.")

    for entry in entries:
        for appliesToTest in appliesToTests:
            if entry.firstChild.nodeValue == appliesToTest:
                return True
    return False


