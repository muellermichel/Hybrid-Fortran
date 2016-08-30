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

from optparse import OptionParser
import struct
import sys, pdb
import math
import traceback

def unpackNextArray(f, readEndianFormat, numOfBytesPerValue, typeSpecifier):
	recordByteLength = unpackNextInteger(f, readEndianFormat)
	if recordByteLength == None:
		return None
	if (recordByteLength % numOfBytesPerValue != 0):
		raise Exception, "Odd record length: %i, modulo %i == 0 expected. Is the file endian correct?" %(recordByteLength, numOfBytesPerValue)
		return None
	recordLength = recordByteLength / numOfBytesPerValue

	data = f.read(recordByteLength)
	if (len(data) != recordByteLength):
		raise Exception, "Could not read %i bytes as expected. Only %i bytes read." %(recordByteLength, len(data))
		return None

	redundantRecordLength = unpackNextInteger(f, readEndianFormat)
	if redundantRecordLength == None:
		raise Exception, "Could not read trailer."
	if (recordByteLength != redundantRecordLength):
		raise Exception, "Header and trailer do not match."
		return None

	dataFormat = '%s%i%s' %(readEndianFormat, recordLength, typeSpecifier)
	try:
		unpacked = struct.unpack(dataFormat, data)
	except Exception as e:
		raise Exception("Error when trying to unpack array using %s format: %s" %(dataFormat, str(e)))
	return unpacked

def unpackNextInteger(f, readEndianFormat):
	data = f.read(4)
	if (len(data) != 4):
		#we have reached the end of the file
		return None

	format = '%si' %(readEndianFormat)
	try:
		unpacked = struct.unpack(format, data)
	except Exception as e:
		raise Exception("Error when trying to unpack integer using %s format: %s" %(format, str(e)))
	return unpacked[0]

def unpackNextFloat(f, readEndianFormat, numOfBytesPerValue, typeSpecifier):
	data = f.read(numOfBytesPerValue)
	if (len(data) != numOfBytesPerValue):
		#we have reached the end of the file
		return None

	format = '%s%s' %(readEndianFormat, typeSpecifier)
	try:
		unpacked = struct.unpack(format, data)
	except Exception as e:
		raise Exception("Error when trying to unpack float using %s format: %s" %(format, str(e)))
	return unpacked[0]

def unpackNextRecord(f, readEndianFormat, numOfBytesPerValue, verbose=False):
	def tentativeUnpack(f, readEndianFormat, numOfBytesPerValue, typeSpecifier, verbose=False):
		currentPosition = f.tell()
		content = None
		eof = False
		try:
			content = unpackNextArray(f, readEndianFormat, numOfBytesPerValue, typeSpecifier)
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
			f.seek(currentPosition)
			content = unpackNextInteger(f, readEndianFormat)
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

		f.seek(currentPosition)
		content = unpackNextFloat(f, readEndianFormat, numOfBytesPerValue, typeSpecifier)
		if content == None:
			return None
		if verbose:
			sys.stderr.write("This record seems to be a float with value %s\n" %(str(content)))
		return [content]

	def valuesAreReasonable(unpacked):
		return all([v > 1E-15 and v < 1E10 for v in unpacked])

	if numOfBytesPerValue != None:
		return tentativeUnpack(
			f,
			readEndianFormat,
			numOfBytesPerValue,
			'd' if numOfBytesPerValue == 8 else 'f',
			verbose
		)

	# at this point we need to guess number of bytes.
	# Let's find out whether the number of records is 0 or 1 with 8 bytes (usually that indicates that it is unlikely to be 8 bytes)
	currentPosition = f.tell()
	unpacked8 = tentativeUnpack(f, readEndianFormat, 8, 'd', verbose)
	if unpacked8 != None and len(unpacked8) > 1:
		return unpacked8
	f.seek(currentPosition)
	unpacked4 = tentativeUnpack(f, readEndianFormat, 4, 'f', verbose)
	reasonable4 = valuesAreReasonable(unpacked4) if unpacked4 else False
	if unpacked8 == None and unpacked4 == None:
		return None
	if unpacked4 != None and len(unpacked4) > 1 and reasonable4:
		return unpacked4
	if unpacked4 == None and unpacked8 != None:
		return unpacked8
	if unpacked8 == None and unpacked4 != None:
		return unpacked4
	if len(unpacked4) == len(unpacked8):
		return unpacked4 #at this point an 4 byte integer is most likely
	if len(unpacked4) == 1 and reasonable4:
		return unpacked4
	return unpacked8

def rootMeanSquareDeviation(tup, tupRef, epsSingle):
	err = 0.0
	newErr = 0.0
	newErrSquare = 0.0
	i = 0
	firstErr = -1
	firstErrVal = 0.0
	firstErrExpected = 0.0
	absoluteErrors = []

	for val in tup:
		expectedVal = tupRef[i]
		newErr = val - expectedVal
		normErr = abs(newErr) / abs(val) if abs(val) > epsSingle else abs(newErr)
		i = i + 1
		absoluteErrors.append(abs(newErr))
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
		elif normErr > epsSingle and firstErr == -1:
			firstErr = i
			firstErrVal = val
			firstErrExpected = expectedVal
		err = err + newErrSquare
	maxErrorIndex = 1 + max(enumerate(absoluteErrors), key=lambda x:x[1])[0] if len(absoluteErrors) > 0 else -1
	mean_or_one = sum(tup) / len(tup) if len(tup) > 0 else 1.0
	return (
		math.sqrt(err) / abs(mean_or_one) if mean_or_one > epsSingle else math.sqrt(err),
		firstErr,
		firstErrVal,
		firstErrExpected,
		maxErrorIndex,
		tup[maxErrorIndex] if maxErrorIndex >= 0 and maxErrorIndex < len(tup) else 0.0,
		tupRef[maxErrorIndex] if maxErrorIndex >= 0 and maxErrorIndex < len(tupRef) else 0.0
	)

def checkIntegrity(tup):
	for index, val in enumerate(tup):
		if math.isnan(val):
			return index, val
		if math.isinf(val):
			return index, val
	return -1, -1

def getEndianFormatString(options, numOfBytesPerValue, fileUsedForAutomaticDetection):
	def getTrialRecordsWithEndianFormat(endianFormat):
		detectionRecords = []
		for recordNum in range(100):
			nextRecord = unpackNextRecord(fileUsedForAutomaticDetection, endianFormat, numOfBytesPerValue, False)
			if nextRecord == None:
				break
			detectionRecords.append(nextRecord)
		return detectionRecords

	if (options.readEndian == "little"):
		return '<'
	if (options.readEndian == "big"):
		return '>'
	fileUsedForAutomaticDetection.seek(0)
	records = getTrialRecordsWithEndianFormat('<')
	fileUsedForAutomaticDetection.seek(0)
	if len(records) == 100 and all([len(r) in [0, 1] for r in records]):
		return '>'
	return '<'

def run_accuracy_test_for_datfile(options, eps, epsSingle):
	numOfBytesPerValue = int(options.bytes) if options.bytes != None else None
	if numOfBytesPerValue != None and numOfBytesPerValue != 4 and numOfBytesPerValue != 8:
		sys.stderr.write("Unsupported number of bytes per value specified.\n")
		sys.exit(2)
	inFile = None
	refFile = None
	try:
		#prepare files
		inFile = open(str(options.inFile),'r')
		if options.refFile != None:
			refFile = open(str(options.refFile),'r')
		else:
			sys.stderr.write("WARNING: No reference file specified - doing some basic checks on the input only\n")
		readEndianFormat = getEndianFormatString(options, numOfBytesPerValue, refFile)
		sys.stderr.write("performing accuracy test with .DAT file, %s bytes per value, %s endian, float\n" %(
			str(numOfBytesPerValue) if numOfBytesPerValue != None else "automatic",
			readEndianFormat
		))
		i = 0
		errorState=False
		while True:
			passedStr = "pass"
			i = i + 1
			unpackedRef = None
			if refFile != None:
				unpackedRef = unpackNextRecord(refFile, readEndianFormat, numOfBytesPerValue, options.verbose)
				if unpackedRef == None:
					break
				if len(unpackedRef) == 0:
					continue
			unpacked = None
			try:
				unpacked = unpackNextRecord(inFile, readEndianFormat, numOfBytesPerValue, options.verbose)
			except(Exception), e:
				sys.stderr.write("Error reading record %i from %s: %s\n" %(i, str(options.inFile), e))
				sys.exit(1)

			if options.verbose and unpacked != None and unpackedRef != None:
				sys.stderr.write("Processing record %i: %i values unpacked for record, %i values unpacked for reference.\n" %(i, len(unpacked), len(unpackedRef)))

			if unpackedRef != None and (unpacked == None or len(unpacked) == 0):
				sys.stderr.write("Error in %s: Record with length %i expected, %s found\n" %(str(options.inFile), len(unpackedRef), str(unpacked)))
				if len(unpackedRef) < 100:
					sys.stderr.write("Expected record: %s\n" %(str(unpackedRef)))
				sys.exit(1)

			if int(options.printNum) > 0:
				print unpacked[0:int(options.printNum)]
			if options.verbose:
				sys.stderr.write("Record %i unpacked, contains %i elements.\n" %(i, len(unpacked)))

			if unpackedRef != None and len(unpacked) != len(unpackedRef):
				sys.stderr.write("Error in %s: Record %i does not have same length as reference. Length: %i, expected: %i\n" \
					%(str(options.inFile), i, len(unpacked), len(unpackedRef)))
				sys.exit(1)
			#analyse unpacked data
			if unpackedRef != None:
				firstInvalidIndex, firstInvalidValue = checkIntegrity(unpackedRef)
				if firstInvalidIndex != -1:
					sys.stderr.write("%s, record %i: WARNING: Invalid Value %s in Reference at %i - cannot analyze\n" %(options.inFile, i, str(firstInvalidValue), firstInvalidIndex))
					continue
			firstInvalidIndex, firstInvalidValue = checkIntegrity(unpacked)
			if firstInvalidIndex != -1:
				errorState=True
				passedStr = "invalid value found: %s; FAIL <-------" %(str(firstInvalidValue))
				firstErr = firstInvalidIndex
				err = -1
				maxErr = -1
				firstErrVal = 0
				expectedVal = 0
				maxErrVal = 0
				maxErrExpectedVal = 0
			elif unpackedRef != None:
				err, firstErr, firstErrVal, expectedVal, maxErr, maxErrVal, maxErrExpectedVal = rootMeanSquareDeviation(unpacked, unpackedRef, epsSingle)
				if firstErr != -1 or err > eps:
					errorState=True
					passedStr = "1st err val: %s; ref: %s; max err val: %s; ref: %s; FAIL" %(firstErrVal, expectedVal, maxErrVal, maxErrExpectedVal)
			sys.stderr.write("%s, rec %i (len%i): RMSE: %e; 1st err idx: %i; max err idx: %i; %s\n" %(
				options.inFile,
				i,
				len(unpacked),
				err,
				firstErr,
				maxErr,
				passedStr
			))
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

def run_accuracy_test_for_netcdf(options, eps, epsSingle):
	def pad_to(text, total_length):
		return "%s%s" %(
			str(text),
			" " * (total_length - len(str(text))) if len(str(text)) < total_length else ""
		)

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

			#analyse NetCDF variable
			ref_array = get_array_from_netcdf_variable(ref_variable)
			if numpy.any(numpy.isnan(ref_array)):
				raise Exception("NaN values present in reference array")
			in_array = get_array_from_netcdf_variable(in_variable)
			if options.slice:
				try:
					in_array = in_array[int(options.slice)] if isinstance(in_array[0], numpy.ndarray) else in_array
					ref_array = ref_array[int(options.slice)] if isinstance(ref_array[0], numpy.ndarray) else ref_array
				except:
					pass
			mean_or_one = numpy.mean(in_array)
			if abs(mean_or_one) < 1E-15:
				mean_or_one = 1.0

			passed_string = None
			result = None
			max_error_index_tuple = None
			max_error = None

			absolute_difference = numpy.abs(in_array - ref_array)
			normalized_error = absolute_difference / mean_or_one
			greater_than_epsilon = normalized_error > epsSingle

			nan_values = numpy.isnan(in_array)
			if numpy.any(nan_values):
				max_error = numpy.NAN
				max_error_index_tuple = numpy.unravel_index(numpy.argmax(numpy.isnan(in_array)), in_array.shape)
			else:
				max_error = numpy.argmax(normalized_error)
				max_error_index_tuple = numpy.unravel_index(max_error, in_array.shape)

			root_mean_square_deviation = numpy.sqrt(numpy.mean((in_array - ref_array)**2))
			root_mean_square_deviation = root_mean_square_deviation / abs(mean_or_one)

			#error found?
			if math.isnan(root_mean_square_deviation):
				result = "FAIL"
				error_found = True
			elif math.isnan(max_error) or numpy.any(greater_than_epsilon) or root_mean_square_deviation > eps:
				result = "FAIL"
				error_found = True
			else:
				result = "pass"

			#print output
			number_of_elements = numpy.prod(in_array.shape)
			if number_of_elements <= 8 and result != "pass":
				passed_string = "input: \n%s\nexpected:\n%s\nerrors found at:%s\n%s" %(in_array, ref_array, greater_than_epsilon, result)
			else:
				passed_string = "MaxErr: %s at: %s of: %s; val: %s; ref: %s; %s" %(
					pad_to(normalized_error[max_error_index_tuple],17),
					pad_to(max_error_index_tuple,18),
					pad_to(in_array.shape,18),
					pad_to(in_array[max_error_index_tuple],20),
					pad_to(ref_array[max_error_index_tuple],20),
					result
				)
			if numpy.count_nonzero(ref_array) == 0:
				passed_string += "; (WARNING:Reference is Zero Matrix!)"
			sys.stderr.write("%s, %s-> nRMSE: %e; %s\n" %(
				options.inFile,
				pad_to(key,20),
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
parser.add_option("-b", "--bytesPerValue", dest="bytes")
parser.add_option("-p", "--printFirstValues", dest="printNum", default="0")
parser.add_option("-r", "--readEndian", dest="readEndian", default="little")
parser.add_option("--netcdf", action="store_true", dest="netcdf")
parser.add_option("--slice", dest="slice", default=None)
parser.add_option("-v", action="store_true", dest="verbose")
parser.add_option("-e", "--epsilon", metavar="EPS", dest="epsilon", help="Throw an error if at any point the error becomes higher than EPS. Defaults to 1E-6.")
(options, args) = parser.parse_args()
eps = 1E-6
epsSingle = 1E-6
if (options.epsilon):
	eps = float(options.epsilon)
if options.netcdf:
	run_accuracy_test_for_netcdf(options, eps, epsSingle)
else:
	run_accuracy_test_for_datfile(options, eps, epsSingle)
