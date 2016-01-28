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
#  Procedure        compareXMLs.py                                     #
#  Comment          Takes two xml files and compares them on node      #
#                   level                                              #
#  Date             2013/01/31                                         #
#  Author           Michel Müller (AOKI Laboratory)                    #
#**********************************************************************#


from xml.dom.minidom import Document
from tools.DomHelper import parseString
from optparse import OptionParser
from tools.GeneralHelper import setupDeferredLogging
import sys, traceback
import logging

def isEqualXML(da, db, ignoreAttributes):
    return isEqualElement(da.documentElement, db.documentElement, ignoreAttributes)

def isEqualElement(a, b, ignoreAttributes):
  if a.tagName!=b.tagName:
      return False

  if len(a.attributes.keys()) != len(b.attributes.keys()):
    return False
  for aatKey, batKey in zip(sorted(a.attributes.keys()), sorted(b.attributes.keys())):
    if aatKey in ignoreAttributes and batKey == aatKey:
      continue
    if aatKey != batKey:
      logging.debug("equality fails for node attributes %s compared to %s; key: %s vs. %s" \
          %(a.toxml(), b.toxml(), aatKey, batKey))
      return False
    if (a.attributes.get(aatKey) == None and b.attributes.get(aatKey) != None) \
    or (a.attributes.get(aatKey) != None and b.attributes.get(aatKey) == None):
      logging.debug("equality fails for node attributes %s compared to %s; one is None, the other not" \
        %(a.toxml(), b.toxml()))
      return False
    if a.attributes.get(aatKey).value != b.attributes.get(aatKey).value:
      logging.debug("equality fails for node attributes %s compared to %s; value: %s vs. %s" \
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

##################### MAIN ##############################
#get all program arguments
parser = OptionParser()
parser.add_option("-i", "--inputXML", dest="inputXML",
                  help="XML file to compare", metavar="XML")
parser.add_option("-r", "--referenceXML", dest="referenceXML",
                  help="XML file to compare to", metavar="XML")
parser.add_option("--ignoreAttributes", dest="ignoreAttributes",
                  help="Attribute names to ignore for comparison, comma separated")
parser.add_option("-d", "--debug", action="store_true", dest="debug",
                  help="show debug print in standard error output")
(options, args) = parser.parse_args()

setupDeferredLogging('preprocessor.log', logging.DEBUG if options.debug else logging.INFO)

if (not options.inputXML):
  logging.error("inputXML option is mandatory. Use '--help' for informations on how to use this module")
  sys.exit(1)

if (not options.referenceXML):
  logging.error("referenceXML option is mandatory. Use '--help' for informations on how to use this module")
  sys.exit(1)

ignoreAttributes = []
if (options.ignoreAttributes):
  ignoreAttributes = options.ignoreAttributes.split(",")

#read in the xmls and compare them
try:
  logging.debug("Starting comparison of %s to %s" %(options.inputXML, options.referenceXML))
  inputXMLFile = open(str(options.inputXML),'r')
  inputXMLData = inputXMLFile.read()
  inputXMLFile.close()
  inputXML = parseString(inputXMLData)
  referenceXMLFile = open(str(options.referenceXML),'r')
  referenceXMLData = referenceXMLFile.read()
  referenceXMLFile.close()
  referenceXML = parseString(referenceXMLData)
except Exception, e:
  logging.debug("Returning False because of exception when opening either %s or %s" %(
    options.inputXML,
    options.referenceXML
  ))
  sys.exit(2)

try:
  result = isEqualXML(inputXML, referenceXML, ignoreAttributes)
  logging.debug("Result of equality test (e.g. are they equal?): %s" %(result))
  if result == True:
    sys.exit(1)
  else:
    sys.exit(2)

except Exception, e:
  logging.critical('Error when comparing xmls: %s%s\n' %(str(e), traceback.format_exc()))
  sys.exit(64)