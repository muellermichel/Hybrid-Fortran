#!/usr/bin/python
# -*- coding: UTF-8 -*-

# Copyright (C) 2013 Michel Müller, Rikagaku Kenkyuujo (RIKEN)

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

def rootMeanSquareDeviation(tup, tupRef, eps):
	err = 0.0
	newErr = 0.0
	newErrSquare = 0.0
	i = 0
	firstErr = -1
	firstErrVal = 0.0
	firstErrExpected = 0.0
	for val in tup:
		expectedVal = tupRef[i]
		newErr = val - expectedVal
		i = i + 1
		try:
			newErrSquare = newErr**2
		except(OverflowError), e:
			firstErr = i
			newErrSquare = sys.float_info.max
			firstErrVal = val
			firstErrExpected = expectedVal
		if math.isnan(newErr) and firstErr == -1:
			firstErr = i
			newErr = 0.0
			newErrSquare = 0.0
			firstErrVal = val
			firstErrExpected = expectedVal
		elif abs(newErr) > eps and firstErr == -1:
			firstErr = i
			firstErrVal = val
			firstErrExpected = expectedVal
		err = err + newErrSquare
	return math.sqrt(err), firstErr, firstErrVal, firstErrExpected

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
parser.add_option("-e", "--epsilon", metavar="EPS", dest="epsilon", help="Throw an error if at any point the error becomes higher than EPS. Defaults to 1E-9.")

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

eps = 1E-9
if (options.epsilon):
	eps = float(options.epsilon)

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

		if unpackedRef == None:
			break;

		unpacked = None
		try:
			unpacked = unpackNextRecord(inFile, readEndianFormat, numOfBytesPerValue)
		except(Exception), e:
			sys.stderr.write("Error reading record %i from %s: %s\n" %(i, str(options.inFile), e))
			sys.exit(1)

		if unpacked == None:
			sys.stderr.write("Error in %s: Record expected, could not load record it\n" %(str(options.inFile)))
			sys.exit(1)

		if len(unpacked) != len(unpackedRef):
			sys.stderr.write("Error in %s: Record %i does not have same length as reference. Length: %i, expected: %i\n" \
				%(str(options.inFile), i, len(unpacked), len(unpackedRef)))
			sys.exit(1)

		#analyse unpacked data
		[err, firstErr, firstErrVal, expectedVal] = rootMeanSquareDeviation(unpacked, unpackedRef, eps)
		errorState=False
		if firstErr != -1 or err > eps:
			errorState=True
			passedStr = "first error value: %s; expected: %s; FAIL <-------" %(firstErrVal, expectedVal)
		sys.stderr.write("%s, record %i: Mean square error: %e; First Error at: %i; %s\n" %(options.inFile, i, err, firstErr, passedStr))

		if options.verbose:
			sys.stderr.write(unpacked + "\n")

		if errorState:
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



