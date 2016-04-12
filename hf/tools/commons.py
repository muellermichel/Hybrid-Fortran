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

import os, sys, re, logging, logging.handlers, atexit, traceback

class UsageError(Exception):
    pass

class HFContextFormatter(logging.Formatter):
    def __init__(self):
        noContextFormat = '%(asctime)s - %(levelname)s - %(message)s'
        contextFormat = '%(asctime)s - %(hfFile)s:%(hfLineNo)s - %(levelname)s - %(message)s'
        logging.Formatter.__init__(self, noContextFormat)
        self.contextFormatter = logging.Formatter(contextFormat)

    def format(self, record):
        if hasattr(record, "hfFile") and hasattr(record, "hfLineNo"):
            return self.contextFormatter.format(record)
        return logging.Formatter.format(self, record)

def stacktrace():
    exc = sys.exc_info()[0]
    stack = traceback.extract_stack()[:-1]  # last one would be full_stack()
    if not exc is None:  # i.e. if an exception is present
        del stack[-1]    # remove call of full_stack, the printed exception
                         # will contain the caught exception caller instead
    return "\n".join([
        "%s:%i(%s)" %(os.path.basename(filename), lineNo, functionName)
        for (filename, lineNo, functionName, _) in stack
    ])

def setupDeferredLogging(filename, logLevel, showDeferredLogging=True):
    logger = logging.getLogger()
    logger.setLevel(logLevel)
    if showDeferredLogging:
        streamFormatter = HFContextFormatter()
        streamhandler = logging.StreamHandler(sys.stderr)
        streamhandler.setLevel(logLevel)
        streamhandler.setFormatter(streamFormatter)
        memoryhandler = logging.handlers.MemoryHandler(
            capacity=1024*100,
            flushLevel=logging.ERROR,
            target=streamhandler
        )
        logger.addHandler(memoryhandler)
        def flush():
            memoryhandler.flush()
        atexit.register(flush)
    logFileFormatter = HFContextFormatter()
    filehandler = logging.FileHandler(filename)
    filehandler.setLevel(logLevel)
    filehandler.setFormatter(logFileFormatter)
    logger.addHandler(filehandler)
    logging.debug("Logger has Initialized")

def progressIndicatorReset(stream):
    stream.write("\n")

def printProgressIndicator(stream, currentlyAtText, currentlyAtNum, totalNum, description):
    stream.write("\r%s: %d%% done.%s" %(
        description,
        round(currentlyAtNum * 100.0/totalNum),
        " Currently processing: " + currentlyAtText if currentlyAtText != "" else ""
    )) #\r returns to beginning of current line
    stream.write("\033[K") #clear rest of current line
    stream.flush()

def stripWhitespace(inputStr):
    match = re.match(r'\s*(.*)\s*', inputStr)
    if not match:
        raise Exception("Whitespace could not be removed from string %s" %(inputStr))
    return match.group(1)

def openFile(file_name, mode):
    """Open a file."""
    try:
        the_file = open(file_name, mode)
    except(IOError), e:
        logging.critical("Unable to open the file %s. Ending program. Error: %s" %(file_name, e))
        sys.exit(1)
    else:
        return the_file

def getDataFromFile(path):
    currFile = openFile(str(path),'rw')
    data = currFile.read()
    currFile.close()
    return data

def prettyprint(obj):
    def convert_to_json(obj):
        from xml.dom.minidom import Document, Node
        import pprint
        output_json = None
        if type(obj) == dict:
            output_json = {}
            for key in obj.keys():
                output_json[str(key)] = convert_to_json(obj[key])
        elif type(obj) == list:
            output_json = []
            for entry in obj:
                output_json.append(convert_to_json(entry))
        elif isinstance(obj, Document) or isinstance(obj, Node):
            output_json = obj.toxml()
        else:
            output_json = unicode(obj)
        return output_json

    import json
    print json.dumps(convert_to_json(obj), sort_keys=True, indent=4, separators=(',', ': '))

def enum(*sequential, **named):
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)

def areIndexesWithinQuotes(stringToSearch):
    #build up a colored list, showing whether an index inside stringToSearch is in quotes or not.
    #nested quotes such as "hello'world' foobar" are not supported!
    quoteSections = re.split(r'''(['"])''', stringToSearch)
    isStringIndexWithinQuote = isStringIndexWithinQuote = [False] * len(stringToSearch)
    if len(quoteSections) < 2:
        pass
    elif (len(quoteSections) - 1) % 2 != 0:
        raise Exception("Unexpected behavior of regex split. Please check your python version.")
    elif (len(quoteSections) - 1) % 4 != 0: #check re.split documentation to see how it works.
        pass
    else:
        quoteSections.reverse()
        currSection = quoteSections.pop()
        index = len(currSection)
        if index > 0:
            isStringIndexWithinQuote[0:index] = [False] * len(currSection)
        while len(quoteSections) > 0:
            #opening quote part
            currSection = quoteSections.pop()
            sectionLength = len(currSection)
            prefIndex = 0
            if index > 0:
                prefIndex = index
            index = index + sectionLength
            if sectionLength != 1:
                raise Exception("Quote begin marker with strange number of characters")
            isStringIndexWithinQuote[prefIndex:index] = [True]

            #inbetween quotes part
            currSection = quoteSections.pop()
            sectionLength = len(currSection)
            prefIndex = index
            index = index + sectionLength
            isStringIndexWithinQuote[prefIndex:index] = [True] * sectionLength

            #closing quote part
            currSection = quoteSections.pop()
            sectionLength = len(currSection)
            prefIndex = index
            index = index + sectionLength
            if sectionLength != 1:
                raise Exception("Quote end marker with strange number of characters")
            isStringIndexWithinQuote[prefIndex:index] = [True]

            #next part that's not within quotes
            currSection = quoteSections.pop()
            sectionLength = len(currSection)
            prefIndex = index
            index = index + sectionLength
            isStringIndexWithinQuote[prefIndex:index] = [False] * sectionLength
        #sanity check
        if index != len(stringToSearch):
            raise Exception("Index at the end of quotes search is %i. Expected: %i" %(index, len(stringToSearch)))
    return isStringIndexWithinQuote

def findRightMostOccurrenceNotInsideQuotes(stringToMatch, stringToSearch, rightStartAt=-1):
    indexesWithinQuotes = areIndexesWithinQuotes(stringToSearch)
    if rightStartAt > 0:
        nextRightStart = rightStartAt
    else:
        nextRightStart = len(stringToSearch)
    blankPos = -1
    for numOfTrys in range(1,101):
        blankPos = stringToSearch[:nextRightStart].rfind(stringToMatch)
        if blankPos <= 0 or not indexesWithinQuotes[blankPos]:
            break
        nextRightStart = blankPos
        blankPos = -1
        if numOfTrys >= 100:
            raise Exception("Could not find the string even after 100 tries.")
    return blankPos

def findLeftMostOccurrenceNotInsideQuotes(stringToMatch, stringToSearch, leftStartAt=-1, filterOutEmbeddings=False):
    indexesWithinQuotes = areIndexesWithinQuotes(stringToSearch)
    nextLeftStart = leftStartAt + 1
    matchIndex = -1
    for numOfTrys in range(1,101):
        if nextLeftStart >= len(stringToMatch):
            break
        matchIndex = stringToSearch[nextLeftStart:].find(stringToMatch)
        if matchIndex < 0:
            break
        matchEndIndex = matchIndex + len(stringToMatch)
        if not indexesWithinQuotes[matchIndex] \
        and (not filterOutEmbeddings or matchIndex < 1 or re.match(r'\W', stringToSearch[matchIndex - 1])) \
        and (not filterOutEmbeddings or len(stringToSearch) <= matchEndIndex + 1 or re.match(r'\W', stringToSearch[matchEndIndex + 1])):
            break
        nextLeftStart = matchIndex + 1
        matchIndex = -1
        if numOfTrys >= 100:
            raise Exception("Could not find the string even after 100 tries.")
    return matchIndex

def splitTextAtLeftMostOccurrence(matchStrings, text):
    def leftMostOccurrenceForName(text, matchString):
        return findLeftMostOccurrenceNotInsideQuotes(matchString, text, filterOutEmbeddings=True), matchString
    if type(matchStrings) in [unicode, str]:
        matchStrings = [matchStrings]
    matchIndex, matchString = -1, None
    for string in matchStrings:
        matchIndex, matchString = leftMostOccurrenceForName(text, string)
        if matchIndex >= 0:
            break
    if matchIndex < 0:
        return text, "", ""
    prefix = ""
    if matchIndex > 0:
        prefix = text[:matchIndex]
    suffix = ""
    if len(text) > matchIndex + len(matchString):
        suffix = text[matchIndex + len(matchString):]
    return prefix, matchString, suffix

class Singleton:
    """
    A non-thread-safe helper class to ease implementing singletons.
    This should be used as a decorator -- not a metaclass -- to the
    class that should be a singleton.

    The decorated class can define one `__init__` function that
    takes only the `self` argument. Other than that, there are
    no restrictions that apply to the decorated class.

    To get the singleton instance, use the `Instance` method. Trying
    to use `__call__` will result in a `TypeError` being raised.

    Limitations: The decorated class cannot be inherited from.
    Source: http://stackoverflow.com/questions/31875/is-there-a-simple-elegant-way-to-define-singletons-in-python
    """

    def __init__(self, decorated):
        self._decorated = decorated

    def Instance(self):
        """
        Returns the singleton instance. Upon its first call, it creates a
        new instance of the decorated class and calls its `__init__` method.
        On all subsequent calls, the already created instance is returned.

        """
        try:
            return self._instance
        except AttributeError:
            self._instance = self._decorated()
            return self._instance

    def __call__(self):
        raise TypeError('Singletons must be accessed through `Instance()`.')

class BracketAnalyzer(object):
    currLevel = 0
    bracketsHaveEverOpened = False
    searchPattern = ""
    openingPattern = ""
    closingPattern = ""
    openingChar = ""
    closingChar = ""

    def __init__(self, openingChar="(", closingChar=")", pass_in_regex_pattern=False):
        self.currLevel = 0
        self.bracketsHaveEverOpened = False
        self.openingChar = openingChar
        self.closingChar = closingChar
        if pass_in_regex_pattern:
            self.searchPattern = re.compile(r"(.*?)(" + openingChar + r"|" + closingChar + r")(.*)", re.IGNORECASE)
            self.openingPattern = re.compile(openingChar, re.IGNORECASE)
            self.closingPattern = re.compile(closingChar, re.IGNORECASE)
        else:
            self.searchPattern = re.compile(r"(.*?)(" + re.escape(openingChar) + r"|" + re.escape(closingChar) + r")(.*)", re.IGNORECASE)
            self.openingPattern = re.compile(re.escape(openingChar), re.IGNORECASE)
            self.closingPattern = re.compile(re.escape(closingChar), re.IGNORECASE)

    @property
    def level(self):
        return self.currLevel

    def splitAfterCharacterOnSameLevelOrClosingBrackets(self, string, char):
        charSearchPattern = re.compile(r"(.*?)(" + re.escape(char) + r"|" + re.escape(self.openingChar) + r"|" + re.escape(self.closingChar) + r")(.*)", re.IGNORECASE)
        charPattern = re.compile(re.escape(char), re.IGNORECASE)
        work = string
        match = charSearchPattern.match(work)
        if not match:
            return "", string, self.currLevel

        startLevel = self.currLevel
        substring = ""
        while match:
            work = match.group(3)
            substring += match.group(1) + match.group(2)
            previousLevel = self.currLevel
            if charPattern.match(match.group(2)) != None and self.bracketsHaveEverOpened == False:
                return substring, work, self.currLevel
            elif self.openingPattern.match(match.group(2)) != None:
                self.bracketsHaveEverOpened = True
                self.currLevel += 1
            elif self.closingPattern.match(match.group(2)) != None:
                self.currLevel -= 1
            elif charPattern.match(match.group(2)) == None:
                raise Exception("Something went wrong in the bracket analysis for string %s - cannot rematch a search character" %(string))
            match = charSearchPattern.match(work)
            if self.currLevel < startLevel or (self.currLevel == startLevel and self.currLevel == previousLevel):
                break
        if self.currLevel <= startLevel:
            return substring, work, self.currLevel
        return "", string, self.currLevel

    def getListOfArgumentsInOpenedBracketsAndRemainder(self, string_without_opening_bracket):
        self.currLevel = 1 #we pass a string where the opening bracket has been removed <-- need to fix level here so it doesn't prematurely split off arguments
        self.bracketsHaveEverOpened = True
        work = string_without_opening_bracket
        arguments = []
        bracketLevel = 0
        while len(work) > 0 and self.currLevel > 0:
            currArgument, work, bracketLevel = self.splitAfterCharacterOnSameLevelOrClosingBrackets(work, ',')
            if bracketLevel > 1 or bracketLevel < 0:
                raise Exception("Something went wrong when trying to split arguments from '%s'" %(work))
            self.currLevel = bracketLevel
            work = work.strip()
            currArgument = currArgument.strip()
            if bracketLevel == 1 and len(work) > 0 and work[0] == ',':
                work = work[1:len(work)]
            if len(currArgument) > 0 and currArgument[-1] == ',' or len(currArgument) > 0 and currArgument[-1] == ')' and bracketLevel < 1:
                currArgument = currArgument[0:len(currArgument)-1]
            if currArgument == "" and bracketLevel == 0:
                break
            if currArgument == "":
                raise Exception("Invalid empty argument. Analyzed string: %s; Bracket Level: %i; Arguments so far: %s; Remainder: %s" %(
                    string_without_opening_bracket,
                    bracketLevel,
                    str(arguments),
                    work
                ))
            arguments.append(currArgument.strip())
        return arguments, work

    #in case the brackets are not closed, the reminder will be an empty string
    #if this happens, this method may be called again, using the same bracket analyzer, with the continued string.
    def splitAfterClosingBrackets(self, string):
        work = string
        match = self.searchPattern.match(work)
        if not match:
            return work, ""

        substring = ""
        while match:
            if self.openingPattern.match(match.group(2)) != None:
                self.bracketsHaveEverOpened = True
                self.currLevel += 1
            elif self.closingPattern.match(match.group(2)) != None:
                if self.currLevel == 0:
                    raise Exception("Closing bracket before opening one.")
                self.currLevel -= 1
            else:
                raise Exception("Something went wrong in the bracket analysis - cannot rematch a search character")
            work = match.group(3)
            substring += match.group(1) + match.group(2)
            if self.currLevel == 0:
                break
            match = self.searchPattern.match(work)
        if self.currLevel == 0:
            return substring, work
        else:
            return string, ""

    #in case the brackets are not closed, the reminder will be an empty string
    #if this happens, this method may be called again, using the same bracket analyzer, with the continued string.
    def getTextWithinBracketsAndRemainder(self, string):
        text, remainder = self.splitAfterClosingBrackets(string)
        #cut away the left and right bracket
        text = text.partition(self.openingChar)[2]
        text = text.rpartition(self.closingChar)[0]
        return text, remainder

    #$$$ some code duplication here with 'splitAfterClosingBrackets' and 'splitAfterCharacterOnSameLevelOrClosingBrackets' - should use a common helper routine
    def currLevelAfterString(self, string):
        work = string
        match = self.searchPattern.match(work)
        while match:
            if self.openingPattern.match(match.group(2)) != None:
                self.bracketsHaveEverOpened = True
                self.currLevel += 1
            elif self.closingPattern.match(match.group(2)) != None:
                if self.currLevel == 0:
                    raise Exception("Closing bracket before opening one.")
                self.currLevel -= 1
            else:
                raise Exception("Something went wrong in the bracket analysis - cannot rematch a search character")
            work = match.group(3)
            match = self.searchPattern.match(work)

        return self.currLevel
