import sys, re, fileinput
import logging
from GeneralHelper import findLeftMostOccurrenceNotInsideQuotes, setupDeferredLogging

openMPLinePattern = re.compile(r'\s*\!\$OMP.*', re.IGNORECASE)
openACCLinePattern = re.compile(r'\s*\!\$ACC.*', re.IGNORECASE)
emptyLinePattern = re.compile(r'(.*?)(?:[\n\r\f\v]+[ \t]*)+(.*)', re.DOTALL)
multiLineContinuationPattern = re.compile(r'(.*?)\s*\&\s+(?:\!?\$?(?:OMP|ACC)?\&)?\s*(.*)', re.DOTALL)

fileInputObject = None
if len(sys.argv) > 1:
	fileInputObject = fileinput.input(sys.argv[1])
else:
	fileInputObject = fileinput.input()

def pre_sanitize_fortran():
	#first pass: strip out commented code (otherwise we could get in trouble when removing line continuations, if there are comments in between)
	noComments = ""
	for line in fileInputObject:
		if openMPLinePattern.match(line) or openACCLinePattern.match(line):
			noComments += line
			continue
		commentIndex = findLeftMostOccurrenceNotInsideQuotes("!", line)
		if commentIndex < 0:
			noComments += line
			continue
		noComments += line[:commentIndex] + "\n"

	#second pass: strip out empty lines (otherwise we could get in trouble when removing line continuations, if there are empty lines in between)
	stripped = ""
	remainder = noComments
	while True:
		emptyLineMatch = emptyLinePattern.match(remainder)
		if not emptyLineMatch:
			stripped += remainder
			break
		stripped += emptyLineMatch.group(1) + "\n"
		remainder = emptyLineMatch.group(2)

	#third pass: remove line continuations
	remainder = stripped
	output = ""
	while True:
		lineContinuationMatch = multiLineContinuationPattern.match(remainder)
		if not lineContinuationMatch:
			output += remainder
			break
		output += lineContinuationMatch.group(1) + " "
		remainder = lineContinuationMatch.group(2)

	print output

setupDeferredLogging('preprocessor.log', logging.INFO)
pre_sanitize_fortran()