# -*- coding: UTF-8 -*-

# Copyright (C) 2015 Michel Müller, Tokyo Institute of Technology

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

def unpackNextArray(file, readEndianFormat, numOfBytesPerValue, typeSpecifier):
	recordByteLength = unpackNextInteger(file, readEndianFormat)
	if recordByteLength == None:
		return None
	if (recordByteLength % numOfBytesPerValue != 0):
		raise Exception, "Odd record length: %i, modulo %i == 0 expected. Is the file endian correct?" %(recordByteLength, numOfBytesPerValue)
		return None
	recordLength = recordByteLength / numOfBytesPerValue

	data = file.read(recordByteLength)
	if (len(data) != recordByteLength):
		raise Exception, "Could not read %i bytes as expected. Only %i bytes read." %(recordByteLength, len(data))
		return None

	redundantRecordLength = unpackNextInteger(file, readEndianFormat)
	if redundantRecordLength == None:
		raise Exception, "Could not read trailer."
	if (recordByteLength != redundantRecordLength):
		raise Exception, "Header and trailer do not match."
		return None

	dataFormat = '%s%i%s' %(readEndianFormat, recordLength, typeSpecifier)
	return struct.unpack(dataFormat, data)

def unpackNextInteger(file, readEndianFormat):
	data = file.read(4)
	if (len(data) != 4):
		#we have reached the end of the file
		return None

	format = '%si' %(readEndianFormat)
	unpacked = struct.unpack(format, data)
	return unpacked[0]

def unpackNextFloat(file, readEndianFormat, numOfBytesPerValue, typeSpecifier):
	data = file.read(numOfBytesPerValue)
	if (len(data) != numOfBytesPerValue):
		#we have reached the end of the file
		return None

	format = '%s%s' %(readEndianFormat, typeSpecifier)
	unpacked = struct.unpack(format, data)
	return unpacked[0]

def unpackNextRecord(file, readEndianFormat, numOfBytesPerValue, typeSpecifier, verbose=False):
	currentPosition = file.tell()
	content = None
	eof = False
	try:
		content = unpackNextArray(file, readEndianFormat, numOfBytesPerValue, typeSpecifier)
		eof = content == None
	except Exception as e:
		if verbose:
			sys.stderr.write("Could not unpack record as array (%s) - trying integer\n" %(str(e)))
	if eof:
		return None
	if content != None:
		if verbose:
			sys.stderr.write("This record seems to be an array of length %i\n" %(len(content)))
		return content

	try:
		file.seek(currentPosition)
		content = unpackNextInteger(file, readEndianFormat)
		eof = content == None
	except Exception as e:
		if verbose:
			sys.stderr.write("Could not unpack record as integer (%s) - trying float as last option\n" %(str(e)))
	if eof:
		return None
	if content != None:
		if verbose:
			sys.stderr.write("This record seems to be an integer with value %i\n" %(content))
		return [content]

	file.seek(currentPosition)
	content = unpackNextFloat(file, readEndianFormat, numOfBytesPerValue, typeSpecifier)
	if content == None:
		return None
	if verbose:
		sys.stderr.write("This record seems to be a float with value %s\n" %(str(content)))
	return [content]

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
			return index, val
		if math.isinf(val):
			return index, val
	return -1, -1

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
	sys.stderr.write("performing accuracy test with .DAT file, %i bytes per value, %s endian, float\n" %(numOfBytesPerValue, readEndianFormat))
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
					unpackedRef = unpackNextRecord(refFile, readEndianFormat, numOfBytesPerValue, typeSpecifier, options.verbose)
				except(Exception), e:
					sys.stderr.write("Error reading record %i from %s: %s\n" %(i, str(options.refFile), e))
					sys.exit(1)
				if unpackedRef == None:
					break;
			unpacked = None
			try:
				unpacked = unpackNextRecord(inFile, readEndianFormat, numOfBytesPerValue, typeSpecifier, options.verbose)
			except(Exception), e:
				sys.stderr.write("Error reading record %i from %s: %s\n" %(i, str(options.inFile), e))
				sys.exit(1)

			if options.verbose and unpacked != None and unpackedRef != None:
				sys.stderr.write("Processing record %i: %i values unpacked for record, %i values unpacked for reference.\n" %(i, len(unpacked), len(unpackedRef)))

			if unpacked == None and unpackedRef != None:
				sys.stderr.write("Error in %s: Record expected, could not load record it\n" %(str(options.inFile)))
				sys.exit(1)
			elif unpacked == None:
				break

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
				firstInvalidIndex, firstInvalidValue = checkIntegrity(unpackedRef)
				if firstInvalidIndex != -1:
					sys.stderr.write("%s, record %i: Invalid Value %s in Reference at %i - cannot analyze\n" %(options.inFile, i, str(firstInvalidValue), firstInvalidIndex))
					continue
				firstInvalidIndex, firstInvalidValue = checkIntegrity(unpacked)
				if firstInvalidIndex != -1:
					errorState=True
					passedStr = "invalid value found: %s; FAIL <-------" %(str(firstInvalidValue))
					firstErr = firstInvalidIndex
					err = -1
				else:
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
	inFile = None
	try:
		inFile = Dataset(options.inFile)
	except Exception as e:
		sys.stderr.write("Error: could not read %s. Error message: %s\n" %(options.inFile, str(e)))
		sys.exit(1)
	refFile = None
	try:
		refFile = Dataset(options.refFile)
	except Exception as e:
		sys.stderr.write("Error: could not read %s. Error message: %s\n" %(options.refFile, str(e)))
		sys.exit(1)
	error_found = False
	for key in inFile.variables.keys():
		try:
			in_array = None
			ref_array = None
			if not key in refFile.variables:
				sys.stderr.write("Error: variable %s not found in reference netcdf file %s\n" %(key, options.refFile))
				error_found = True
				continue
			in_variable = inFile.variables[key]
			if in_variable.dtype.kind == 'S':
				sys.stderr.write("Skipping variable %s with data type %s\n" %(key, in_variable.dtype))
				continue
			ref_variable = refFile.variables[key]
			if in_variable.dtype != ref_variable.dtype:
				sys.stderr.write("Error: variable %s has different datatypes - infile: %s, reference: %s\n" %(key, in_variable.dtype, ref_variable.dtype))
				error_found = True
				continue
			try:
				shape_comparison = numpy.equal(in_variable.shape, ref_variable.shape)
			except Exception:
				sys.stderr.write("Error: variable %s's shape (%s) could not be compared to the reference shape (%s)\n" %(key, str(in_variable.shape), ref_variable.shape))
				error_found = True
				continue
			if not numpy.all(shape_comparison):
				sys.stderr.write("Error: variable %s has different shapes - infile: %s, reference: %s\n" %(key, in_variable.shape, ref_variable.shape))
				error_found = True
				continue
			in_array = get_array_from_netcdf_variable(in_variable)
			ref_array = get_array_from_netcdf_variable(ref_variable)
			absolute_difference = numpy.abs(in_array - ref_array)
			greater_than_epsilon = absolute_difference > eps
			passed_string = "pass"
			if numpy.count_nonzero(ref_array) == 0:
				passed_string += "(WARNING:Reference is Zero Matrix!)"
			root_mean_square_deviation = numpy.sqrt(numpy.mean((in_array - ref_array)**2))
			if math.isnan(root_mean_square_deviation):
				passed_string = "FAIL <-------"
				error_found = True
			elif numpy.any(greater_than_epsilon):
				error_found = True
				# first_error = numpy.unravel_index(numpy.argmax(greater_than_epsilon), greater_than_epsilon.shape)[0]
				number_of_elements = numpy.prod(in_array.shape)
				if number_of_elements <= 8:
					passed_string = "input: \n%s\nexpected:\n%s\nerrors found at:%s\nFAIL <-------" %(in_array, ref_array, greater_than_epsilon)
				else:
					first_occurrence = numpy.argmax(greater_than_epsilon==True)
					first_occurrence_index_tuple = numpy.unravel_index(first_occurrence, in_array.shape)
					first_err_val = in_array[first_occurrence_index_tuple]
					expected_val = ref_array[first_occurrence_index_tuple]
					passed_string = "first error at:%s; first error value: %s; expected: %s; FAIL <-------" %(first_occurrence, first_err_val, expected_val)
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
parser.add_option("-b", "--bytesPerValue", dest="bytes", default="8")
parser.add_option("-p", "--printFirstValues", dest="printNum", default="0")
parser.add_option("-r", "--readEndian", dest="readEndian", default="little")
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