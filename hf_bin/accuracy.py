#!/usr/bin/python
# -*- coding: UTF-8 -*-

from optparse import OptionParser
import struct
import sys
import math
import traceback

def unpackNextRecord(file, readEndianFormat, numOfBytesPerValue):
	header = file.read(4)
	if (len(header) != 4):
		#we have reached the end of the file
		return None

	headerFormat = '%si' %(readEndianFormat)
	headerUnpacked = struct.unpack(headerFormat, header)
	recordByteLength = headerUnpacked[0]
	if (recordByteLength % numOfBytesPerValue != 0):
		raise Exception, "Odd record length."
		return None
	recordLength = recordByteLength / numOfBytesPerValue

	data = file.read(recordByteLength)
	if (len(data) != recordByteLength):
		raise Exception, "Could not read %i bytes as expected. Only %i bytes read." %(recordByteLength, len(data))
		return None

	trailer = file.read(4)
	if (len(trailer) != 4):
		raise Exception, "Could not read trailer."
		return None
	trailerUnpacked = struct.unpack(headerFormat, trailer)
	redundantRecordLength = trailerUnpacked[0]
	if (recordByteLength != redundantRecordLength):
		raise Exception, "Header and trailer do not match."
		return None

	dataFormat = '%s%i%s' %(readEndianFormat, recordLength, typeSpecifier)
	return struct.unpack(dataFormat, data)

def rootMeanSquareDeviation(tup, tupRef):
	err = 0.0
	i = 0
	firstErr = -1
	firstErrVal = 0.0
	expectedVal = 0.0
	for val in tup:
		try:
			newErr = (val - tupRef[i])**2
		except(OverflowError), e:
			firstErr = i
			newErr = 0.0
			firstErrVal = val
			expectedVal = tupRef[i]
		i = i + 1
		if math.isnan(newErr) and firstErr == -1:
			firstErr = i
			newErr = 0.0
			firstErrVal = val
			expectedVal = tupRef[i]
		elif newErr > 1E-08 and firstErr == -1:
			firstErr = i
			firstErrVal = val
			expectedVal = tupRef[i]
		err = err + newErr

	return math.sqrt(err), firstErr, firstErrVal, expectedVal

##################### MAIN ##############################
#get all program arguments
parser = OptionParser()
parser.add_option("-f", "--file", dest="inFile",
                  help="read from FILE", metavar="FILE", default="in.dat")
parser.add_option("--reference", dest="refFile",
                  help="reference FILE", metavar="FILE", default="ref.dat")
parser.add_option("-b", "--bytesPerValue", dest="bytes", default="4")
parser.add_option("-r", "--readEndian", dest="readEndian", default="big")
parser.add_option("-v", action="store_true", dest="verbose")

(options, args) = parser.parse_args()

numOfBytesPerValue = int(options.bytes)
if (numOfBytesPerValue != 4 and numOfBytesPerValue != 8):
	sys.stderr.write("Unsupported number of bytes per value specified.\n")
	sys.exit(2)
typeSpecifier = 'f'
if (numOfBytesPerValue == 8):
	typeSpecifier = 'd'

readEndianFormat = '>'
if (options.readEndian == "little"):
	readEndianFormat = '<'

inFile = None
refFile = None
try:
	#prepare files
	inFile = open(str(options.inFile),'r')
	refFile = open(str(options.refFile),'r')

	i = 0
	while True:
		passedStr = "pass"
		i = i + 1
		unpackedRef = None
		try:
			unpackedRef = unpackNextRecord(refFile, readEndianFormat, numOfBytesPerValue)
		except(Exception), e:
			sys.stderr.write("Error reading record %i from %s: %s\n" %(i, str(options.refFile), e))
			sys.exit(1)

		if (unpackedRef == None):
			break;

		unpacked = None
		try:
			unpacked = unpackNextRecord(inFile, readEndianFormat, numOfBytesPerValue)
		except(Exception), e:
			sys.stderr.write("Error reading record %i from %s: %s\n" %(i, str(options.inFile), e))
			sys.exit(1)

		if (unpacked == None):
			sys.stderr.write("Error in %s: Record expected, could not load record it\n" %(str(options.inFile)))
			sys.exit(1)

		if (len(unpacked) != len(unpackedRef)):
			sys.stderr.write("Error in %s: Record %i does not have same length as reference. Length: %i, expected: %i\n" \
				%(str(options.inFile), i, len(unpacked), len(unpackedRef)))
			sys.exit(1)

		#analyse unpacked data
		[err, firstErr, firstErrVal, expectedVal] = rootMeanSquareDeviation(unpacked, unpackedRef)
		if (firstErr != -1 or err > 1E-8):
			passedStr = "first error value: %s; expected: %s; FAIL <-------" %(firstErrVal, expectedVal)
		sys.stderr.write("%s, record %i: Mean square error: %e; First Error at: %i; %s\n" %(options.inFile, i, err, firstErr, passedStr))

		if (options.verbose):
			sys.stderr.write(unpacked + "\n")

		if (firstErr != -1):
			sys.exit(1)

except(Exception), e:
	sys.stderr.write("Error: %s\n%s\n" %(e, traceback.format_exc()))
	sys.exit(1)

finally:
	#cleanup
	if inFile != None:
		inFile.close()

	if refFile != None:
		refFile.close()



