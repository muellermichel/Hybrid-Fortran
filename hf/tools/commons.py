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
import collections
import time
import functools
from UserDict import DictMixin

class OrderedDict(dict, DictMixin):

    def __init__(self, *args, **kwds):
        if len(args) > 1:
            raise TypeError('expected at most 1 arguments, got %d' % len(args))
        try:
            self.__end
        except AttributeError:
            self.clear()
        self.update(*args, **kwds)

    def clear(self):
        self.__end = end = []
        end += [None, end, end]         # sentinel node for doubly linked list
        self.__map = {}                 # key --> [key, prev, next]
        dict.clear(self)

    def __setitem__(self, key, value):
        if key not in self:
            end = self.__end
            curr = end[1]
            curr[2] = end[1] = self.__map[key] = [key, curr, end]
        dict.__setitem__(self, key, value)

    def __delitem__(self, key):
        dict.__delitem__(self, key)
        key, prev, next = self.__map.pop(key)
        prev[2] = next
        next[1] = prev

    def __iter__(self):
        end = self.__end
        curr = end[2]
        while curr is not end:
            yield curr[0]
            curr = curr[2]

    def __reversed__(self):
        end = self.__end
        curr = end[1]
        while curr is not end:
            yield curr[0]
            curr = curr[1]

    def popitem(self, last=True):
        if not self:
            raise KeyError('dictionary is empty')
        if last:
            key = reversed(self).next()
        else:
            key = iter(self).next()
        value = self.pop(key)
        return key, value

    def __reduce__(self):
        items = [[k, self[k]] for k in self]
        tmp = self.__map, self.__end
        del self.__map, self.__end
        inst_dict = vars(self).copy()
        self.__map, self.__end = tmp
        if inst_dict:
            return (self.__class__, (items,), inst_dict)
        return self.__class__, (items,)

    def keys(self):
        return list(self)

    setdefault = DictMixin.setdefault
    update = DictMixin.update
    pop = DictMixin.pop
    values = DictMixin.values
    items = DictMixin.items
    iterkeys = DictMixin.iterkeys
    itervalues = DictMixin.itervalues
    iteritems = DictMixin.iteritems

    def __repr__(self):
        if not self:
            return '%s()' % (self.__class__.__name__,)
        return '%s(%r)' % (self.__class__.__name__, self.items())

    def copy(self):
        return self.__class__(self)

    @classmethod
    def fromkeys(cls, iterable, value=None):
        d = cls()
        for key in iterable:
            d[key] = value
        return d

    def __eq__(self, other):
        if isinstance(other, OrderedDict):
            if len(self) != len(other):
                return False
            for p, q in  zip(self.items(), other.items()):
                if p != q:
                    return False
            return True
        return dict.__eq__(self, other)

    def __ne__(self, other):
        return not self == other

# Raymond Hettinger's immutable dict from
# http://stackoverflow.com/questions/9997176/immutable-dictionary-only-use-as-a-key-for-another-dictionary
class ImmutableDict(collections.Mapping):
    def __init__(self, somedict):
        self._dict = dict(somedict)
        self._hash = None

    def __getitem__(self, key):
        return self._dict[key]

    def __len__(self):
        return len(self._dict)

    def __iter__(self):
        return iter(self._dict)

    def __hash__(self):
        if self._hash is None:
            self._hash = hash(frozenset(self._dict.items()))
        return self._hash

    def __eq__(self, other):
        return self._dict == other._dict

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

# from http://stackoverflow.com/questions/11815873/memoization-library-for-python-2-7
def lru_cache(maxsize = 255, timeout = None):
    """lru_cache(maxsize = 255, timeout = None) --> returns a decorator which returns an instance (a descriptor).

        Purpose         - This decorator factory will wrap a function / instance method and will supply a caching mechanism to the function.
                            For every given input params it will store the result in a queue of maxsize size, and will return a cached ret_val
                            if the same parameters are passed.

        Params          - maxsize - int, the cache size limit, anything added above that will delete the first values enterred (FIFO).
                            This size is per instance, thus 1000 instances with maxsize of 255, will contain at max 255K elements.
                        - timeout - int / float / None, every n seconds the cache is deleted, regardless of usage. If None - cache will never be refreshed.

        Notes           - If an instance method is wrapped, each instance will have it's own cache and it's own timeout.
                        - The wrapped function will have a cache_clear variable inserted into it and may be called to clear it's specific cache.
                        - The wrapped function will maintain the original function's docstring and name (wraps)
                        - The type of the wrapped function will no longer be that of a function but either an instance of _LRU_Cache_class or a functool.partial type.

        On Error        - No error handling is done, in case an exception is raised - it will permeate up.
    """

    class _LRU_Cache_class(object):
        def __init__(self, input_func, max_size, timeout):
            self._input_func        = input_func
            self._max_size          = max_size
            self._timeout           = timeout

            # This will store the cache for this function, format - {caller1 : [OrderedDict1, last_refresh_time1], caller2 : [OrderedDict2, last_refresh_time2]}.
            #   In case of an instance method - the caller is the instance, in case called from a regular function - the caller is None.
            self._caches_dict        = {}

        def cache_clear(self, caller = None):
            # Remove the cache for the caller, only if exists:
            if caller in self._caches_dict:
                del self._caches_dict[caller]
                self._caches_dict[caller] = [collections.OrderedDict(), time.time()]

        def __get__(self, obj, objtype):
            """ Called for instance methods """
            return_func = functools.partial(self._cache_wrapper, obj)
            return_func.cache_clear = functools.partial(self.cache_clear, obj)
            # Return the wrapped function and wraps it to maintain the docstring and the name of the original function:
            return functools.wraps(self._input_func)(return_func)

        def __call__(self, *args, **kwargs):
            """ Called for regular functions """
            return self._cache_wrapper(None, *args, **kwargs)
        # Set the cache_clear function in the __call__ operator:
        __call__.cache_clear = cache_clear


        def _cache_wrapper(self, caller, *args, **kwargs):
            # Create a unique key including the types (in order to differentiate between 1 and '1'):
            kwargs_key = "".join(map(lambda x : str(x) + str(type(kwargs[x])) + str(kwargs[x]), sorted(kwargs)))
            key = "".join(map(lambda x : str(type(x)) + str(x) , args)) + kwargs_key

            # Check if caller exists, if not create one:
            if caller not in self._caches_dict:
                self._caches_dict[caller] = [collections.OrderedDict(), time.time()]
            else:
                # Validate in case the refresh time has passed:
                if self._timeout != None:
                    if time.time() - self._caches_dict[caller][1] > self._timeout:
                        self.cache_clear(caller)

            # Check if the key exists, if so - return it:
            cur_caller_cache_dict = self._caches_dict[caller][0]
            if key in cur_caller_cache_dict:
                return cur_caller_cache_dict[key]

            # Validate we didn't exceed the max_size:
            if len(cur_caller_cache_dict) >= self._max_size:
                # Delete the first item in the dict:
                cur_caller_cache_dict.popitem(False)

            # Call the function and store the data in the cache (call it with the caller in case it's an instance function - Ternary condition):
            cur_caller_cache_dict[key] = self._input_func(caller, *args, **kwargs) if caller != None else self._input_func(*args, **kwargs)
            return cur_caller_cache_dict[key]


    # Return the decorator wrapping the class (also wraps the instance to maintain the docstring and the name of the original function):
    return (lambda input_func : functools.wraps(input_func)(_LRU_Cache_class(input_func, maxsize, timeout)))

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

def printProgressIndicator(stream, fileName, lineNum, totalNum, description):
    stream.write("\r%s: %d%% done.%s" %(
        description,
        round(lineNum * 100.0/totalNum),
        " Currently processing: " + os.path.basename(fileName) if fileName != "" else ""
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
        if nextLeftStart >= len(stringToSearch):
            break
        currSlice = stringToSearch[nextLeftStart:]
        matchIndex = currSlice.find(stringToMatch)
        if matchIndex < 0:
            break
        matchEndIndex = matchIndex + len(stringToMatch)
        if not indexesWithinQuotes[nextLeftStart:][matchIndex] \
        and (not filterOutEmbeddings or nextLeftStart + matchIndex < 1 or re.match(r'\W', stringToSearch[nextLeftStart + matchIndex - 1])) \
        and (not filterOutEmbeddings or len(stringToSearch) <= nextLeftStart + matchEndIndex or re.match(r'\W', stringToSearch[nextLeftStart + matchEndIndex])):
            break
        nextLeftStart += matchIndex + 1
        matchIndex = -1
        if numOfTrys >= 100:
            raise Exception("Could not find the string even after 100 tries.")
    return matchIndex + nextLeftStart if matchIndex >= 0 else -1

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

def splitIntoComponentsAndRemainder(string):
    if string.strip() == "":
        return [], ""
    analyzer = BracketAnalyzer()
    splitted = string.split(',')
    currComponent = ""
    components = []
    for index, part in enumerate(splitted):
        currComponent += part
        if analyzer.currLevelAfterString(part) == 0:
            components.append(currComponent.strip())
            currComponent = ""
            continue
        if index < len(splitted) - 1:
            currComponent += ','
    remainder = currComponent
    if len(components) > 0:
        separated = re.split(r'([^\w\,\:\+\-\*\/\(\)])', components[-1])
        analyzer = BracketAnalyzer()
        lastComponent = ""
        for index, part in enumerate(separated):
            lastComponent += part
            lookAheadCharacter = None
            if index < len(separated) - 1:
                for lookAheadIndex in range(index + 1, len(separated)):
                    nextPart = separated[lookAheadIndex].strip()
                    if len(nextPart) > 0:
                        lookAheadCharacter = nextPart[0]
                        break
            bracketLevelAfterPart = analyzer.currLevelAfterString(part)
            if lookAheadCharacter != '(' \
            and bracketLevelAfterPart == 0:
                components[-1] = lastComponent.strip()
                if len(separated) > index + 1:
                    remainder = ''.join(separated[index + 1:])
                else:
                    remainder = ""
                break
    return components, remainder.strip()

def getComponentNameAndBracketContent(component):
    leftBracketIndex = component.find('(')
    if leftBracketIndex < 0:
        return component.strip(), None
    if leftBracketIndex == 0:
        raise Exception("invalid component: %s" %(component))
    rightBracketIndex = component.rfind(')')
    if rightBracketIndex < 0 or rightBracketIndex <= leftBracketIndex + 1:
        raise Exception("invalid component: %s" %(component))
    return component[:leftBracketIndex].strip(), component[leftBracketIndex+1:rightBracketIndex]

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