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

##################### MAIN ##############################
#get all program arguments
parser = OptionParser()
parser.add_option("-f", "--file", dest="inFile",
                  help="read from FILE", metavar="FILE", default="in.dat")
parser.add_option("-o", "--out", dest="outFile",
                  help="write to FILE", metavar="FILE", default="out.dat")
parser.add_option("--min", dest="min",
                  help="test all read values for MIN", metavar="MIN", default="-1E10")
parser.add_option("--max", dest="max",
                  help="test all read values for MAX", metavar="MAX", default="1E10")
parser.add_option("-b", "--bytesPerValue", dest="bytes", default="4")
parser.add_option("-p", "--numberOfRecordsToPrint", dest="printRecords", default="0")
parser.add_option("-r", "--readEndian", dest="readEndian", default="big")
parser.add_option("-w", "--writeEndian", dest="writeEndian", default="little")
(options, args) = parser.parse_args()

#initialise according to input parameters
min = eval(options.min)
max = eval(options.max)
numOfBytesPerValue = int(options.bytes)
if (numOfBytesPerValue != 4 and numOfBytesPerValue != 8):
	print "Unsupported number of bytes per value specified."
	sys.exit()
readEndianFormat = '>'
writeEndianFormat = '<'
if (options.readEndian == "little"):
	readEndianFormat = '<'
if (options.writeEndian == "big"):
	readEndianFormat = '>'

typeSpecifier = 'f'
if (numOfBytesPerValue == 8):
	typeSpecifier = 'd'

numOfRecordsToPrint = int(options.printRecords)

i = 0
try:
	#prepare files
	inFile = open(str(options.inFile),'r')
	outFile = open(str(options.outFile),'w')

	printedRecords = 0
	while True:
		header = inFile.read(4)
		if (len(header) != 4):
			#we have reached the end of the file
			break
		i = i + 1
		print "preparing record %i" %(i)
		headerFormat = '%si' %(readEndianFormat)
		headerUnpacked = struct.unpack(headerFormat, header)
		recordByteLength = headerUnpacked[0]
		if (recordByteLength % numOfBytesPerValue != 0):
			print "Odd record length at record %i. Abort." %(i)
			sys.exit()
		print "record byte length: %i" %(recordByteLength)
		recordLength = recordByteLength / numOfBytesPerValue
		print "record length: %i" %(recordLength)

		data = inFile.read(recordByteLength)
		if (len(data) != recordByteLength):
			print "Could not read %i bytes at record %i. Abort." %(recordByteLength, i)
			sys.exit()

		dataFormat = '%s%i%s' %(readEndianFormat, recordLength, typeSpecifier)
		unpacked = struct.unpack(dataFormat, data)

		#analyse unpacked data (to make sure we have read something useful). Print some values if specified.
		if (printedRecords < numOfRecordsToPrint):
			print "record %i: %s" %(i, unpacked)
		j = 0
		for val in unpacked:
			j = j + 1
			if (val > min and val < max):
				continue
			else:
				print "value %i in record %i is out of the specified bounds: %d" %(j, i, val)
				sys.exit()

		trailer = inFile.read(4)
		if (len(trailer) != 4):
			print "Could not read trailer at record %i. Abort." %(i)
			sys.exit()
		trailerUnpacked = struct.unpack(headerFormat, trailer)
		redundantRecordLength = trailerUnpacked[0]
		if (recordByteLength != redundantRecordLength):
			print "Header and trailer do not match at record %i. Abort." %(i)
			sys.exit()

		repackedRecordFormat = '%si%i%si' %(writeEndianFormat, recordLength, typeSpecifier)
		print "write format: %s" %(repackedRecordFormat)
		unpackedWithHeaderAndTrailer = (recordByteLength,) + unpacked + (recordByteLength,)
		repacked = struct.pack(repackedRecordFormat, *unpackedWithHeaderAndTrailer)
		outFile.write(repacked)
		print "record %i written, containing %i bytes" %(i, redundantRecordLength)

finally:
	print "number of records read: %i" %(i)
	#cleanup
	inFile.close()
	outFile.close()



