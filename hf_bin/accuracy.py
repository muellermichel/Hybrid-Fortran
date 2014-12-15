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

from optparse import OptionParser
import struct
import sys, pdb
import math
import traceback

def unpackNextRecord(file, readEndianFormat, numOfBytesPerValue, typeSpecifier):
	header = file.read(4)
	if (len(header) != 4):
		#we have reached the end of the file
		return None

	headerFormat = '%si' %(readEndianFormat)
	headerUnpacked = struct.unpack(headerFormat, header)
	recordByteLength = headerUnpacked[0]
	if (recordByteLength % numOfBytesPerValue != 0):
		raise Exception, "Odd record length: %i, modulo %i == 0 expected. Is the file endian correct?" %(recordByteLength, numOfBytesPerValue)
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

def checkIntegrity(tup):
	for index, val in enumerate(tup):
		if math.isnan(val):
			raise Exception("value at index %i in input file is not a number." %(index))
		if math.isinf(val):
			raise Exception("value at index %i in input file is an infinite number." %(index))

def run_accuracy_test_for_datfile(options, eps):
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
		if options.refFile != None:
			refFile = open(str(options.refFile),'r')
		else:
			sys.stderr.write("WARNING: No reference file specified - doing some basic checks on the input only\n")
		i = 0
		errorState=False
		while True:
			passedStr = "pass"
			i = i + 1
			unpackedRef = None
			if refFile != None:
				try:
					unpackedRef = unpackNextRecord(refFile, readEndianFormat, numOfBytesPerValue, typeSpecifier)
				except(Exception), e:
					sys.stderr.write("Error reading record %i from %s: %s\n" %(i, str(options.refFile), e))
					sys.exit(1)
				if unpackedRef == None:
					break;
			unpacked = None
			try:
				unpacked = unpackNextRecord(inFile, readEndianFormat, numOfBytesPerValue, typeSpecifier)
			except(Exception), e:
				sys.stderr.write("Error reading record %i from %s: %s\n" %(i, str(options.inFile), e))
				sys.exit(1)

			if unpacked == None and unpackedRef != None:
				sys.stderr.write("Error in %s: Record expected, could not load record it\n" %(str(options.inFile)))
				sys.exit(1)
			elif unpacked == None:
				break

			checkIntegrity(unpacked)
			if int(options.printNum) > 0:
				print unpacked[0:int(options.printNum)]
			if options.verbose:
				sys.stderr.write("Record %i unpacked, contains %i elements.\n" %(i, len(unpacked)))

			if unpackedRef != None:
				if len(unpacked) != len(unpackedRef):
					sys.stderr.write("Error in %s: Record %i does not have same length as reference. Length: %i, expected: %i\n" \
						%(str(options.inFile), i, len(unpacked), len(unpackedRef)))
					sys.exit(1)
				#analyse unpacked data
				[err, firstErr, firstErrVal, expectedVal] = rootMeanSquareDeviation(unpacked, unpackedRef, eps)
				if firstErr != -1 or err > eps:
					errorState=True
					passedStr = "first error value: %s; expected: %s; FAIL <-------" %(firstErrVal, expectedVal)
				sys.stderr.write("%s, record %i: Mean square error: %e; First Error at: %i; %s\n" %(options.inFile, i, err, firstErr, passedStr))
	except(Exception), e:
		sys.stderr.write("Error: %s\n" %(e))
		sys.exit(1)
	finally:
		#cleanup
		if inFile != None:
			inFile.close()
		if refFile != None:
			refFile.close()
	if errorState:
		sys.exit(1)

def run_accuracy_test_for_netcdf(options, eps):
	def get_array_from_netcdf_variable(netcdf_variable):
		num_of_dimensions = len(netcdf_variable.dimensions)
		if num_of_dimensions == 1:
			return netcdf_variable[:]
		if num_of_dimensions == 2:
			return netcdf_variable[:, :]
		if num_of_dimensions == 3:
			return netcdf_variable[:, :, :]
		if num_of_dimensions == 4:
			return netcdf_variable[:, :, :, :]
		if num_of_dimensions == 5:
			return netcdf_variable[:, :, :, :, :]

	from netCDF4 import Dataset
	import numpy
	inFile = Dataset(options.inFile)
	refFile = Dataset(options.refFile)
	error_found = False
	for key in inFile.variables.keys():
		try:
			in_array = None
			ref_array = None
			if not key in refFile.variables:
				sys.stderr.write("Error: variable %s not found in reference netcdf file %s\n" %(key, options.refFile))
				sys.exit(1)
			in_variable = inFile.variables[key]
			if in_variable.dtype.kind == 'S':
				sys.stderr.write("Skipping variable %s with data type %s\n" %(key, in_variable.dtype))
				continue
			ref_variable = refFile.variables[key]
			if in_variable.dtype != ref_variable.dtype:
				sys.stderr.write("Error: variable %s has different datatypes - infile: %s, reference: %s\n" %(key, in_variable.dtype, ref_variable.dtype))
				sys.exit(1)
			shape_comparison = numpy.equal(in_variable.shape, ref_variable.shape)
			if not numpy.all(shape_comparison):
				sys.stderr.write("Error: variable %s has different shapes - infile: %s, reference: %s\n" %(key, in_variable.shape, ref_variable.shape))
				sys.exit(1)
			in_array = get_array_from_netcdf_variable(in_variable)
			ref_array = get_array_from_netcdf_variable(ref_variable)
			absolute_difference = numpy.abs(in_array - ref_array)
			greater_than_epsilon = absolute_difference > eps
			passed_string = "pass"
			if numpy.count_nonzero(ref_array) == 0:
				passed_string += "(WARNING:Reference is Zero Matrix!)"
			root_mean_square_deviation = numpy.sqrt(numpy.mean((in_array - ref_array)**2))
			if numpy.any(greater_than_epsilon):
				error_found = True
				# first_error = numpy.unravel_index(numpy.argmax(greater_than_epsilon), greater_than_epsilon.shape)[0]
				number_of_elements = numpy.prod(in_array.shape)
				if number_of_elements <= 8:
					passed_string = "input: \n%s\nexpected:\n%s\nerrors found at:%s\nFAIL <-------" %(in_array, ref_array, greater_than_epsilon)
				else:
					first_occurrence = numpy.argmax(greater_than_epsilon==True)
					passed_string = "first error at:%s; FAIL <-------" %(first_occurrence)
			elif root_mean_square_deviation > eps:
				error_found = True
				passed_string = "FAIL <-------"
			sys.stderr.write("%s, variable %s: Mean square error: %e; %s\n" %(
				options.inFile,
				key,
				root_mean_square_deviation,
				passed_string
			))
		except Exception as e:
			message = "Variable %s\nError Message: %s\n%s\ninarray:%s\nrefarray:%s" %(key, str(e), traceback.format_tb(sys.exc_info()[2]), str(in_array), str(ref_array))
			e.args = (message,)+e.args[1:]
			raise e
	if error_found:
		sys.exit(1)

##################### MAIN ##############################
#get all program arguments
parser = OptionParser()
parser.add_option("-f", "--file", dest="inFile",
                  help="read from FILE", metavar="FILE", default="in.dat")
parser.add_option("--reference", dest="refFile",
                  help="reference FILE", metavar="FILE", default=None)
parser.add_option("-b", "--bytesPerValue", dest="bytes", default="4")
parser.add_option("-p", "--printFirstValues", dest="printNum", default="0")
parser.add_option("-r", "--readEndian", dest="readEndian", default="big")
parser.add_option("--netcdf", action="store_true", dest="netcdf")
parser.add_option("-v", action="store_true", dest="verbose")
parser.add_option("-e", "--epsilon", metavar="EPS", dest="epsilon", help="Throw an error if at any point the error becomes higher than EPS. Defaults to 1E-9.")
(options, args) = parser.parse_args()
eps = 1E-9
if (options.epsilon):
	eps = float(options.epsilon)
if options.netcdf:
	run_accuracy_test_for_netcdf(options, eps)
else:
	run_accuracy_test_for_datfile(options, eps)