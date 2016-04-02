import math, numpy, os, struct
from optparse import OptionParser
import matplotlib.pyplot as pyplot
import matplotlib

def dir_entries(dir_path='', subdir=False, *args):
	'''Return a list of file names found in directory 'dir_path'
	If 'subdir' is True, recursively access subdirectories under 'dir_path'.
	Additional arguments, if any, are file extensions to match filenames. Matched
		file names are added to the list.
	If there are no additional arguments, all files found in the directory are
		added to the list.
	Example usage: fileList = dirEntries(r'H:\TEMP', False, 'txt', 'py')
		Only files with 'txt' and 'py' extensions will be added to the list.
	Example usage: fileList = dirEntries(r'H:\TEMP', True)
		All files and all the files in subdirectories under H:\TEMP will be added
		to the list.
	'''
	dir_path = os.getcwd() + os.sep + dir_path

	fileList = []
	for file in os.listdir(dir_path):
		dirfile = os.path.join(dir_path, file)
		if os.path.isfile(dirfile):
			if not "DS_Store" in dirfile:
				if not args:
					fileList.append(dirfile)
				else:
					if os.path.splitext(dirfile)[1][1:] in args:
						fileList.append(dirfile)
		# recursively access file names in subdirectories
		elif os.path.isdir(dirfile) and subdir:
			fileList.extend(dir_entries(dirfile, subdir, *args))
	return fileList

def get_unpacked_data(file, readEndianFormat='<', numOfBytesPerValue=8, typeSpecifier='d'):
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

parser = OptionParser()
parser.add_option("--nx", dest="nx")
parser.add_option("--ny", dest="ny")
parser.add_option("--nz", dest="nz")
(options, args) = parser.parse_args()

nx = int(options.nx) if options.nx != None else 200
ny = int(options.ny) if options.ny != None else 200
nz = int(options.nz) if options.nz != None else 200

j_axis_cut = int(ny/2)

# define the grid over which the function should be plotted (xx and yy are matrices)
xx, zz = numpy.meshgrid(
	numpy.linspace(0, nz, nz),
	numpy.linspace(0, nx+2, nx+2)
)

if not os.path.exists("./plot"):
	os.makedirs("./plot")

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

for filename in dir_entries('./out'):
	dataset_name = os.path.basename(filename)
	print "unpacking %s" %(dataset_name)
	data = get_unpacked_data(open(filename, 'r'))
	data_length = (nx + 2) * (ny + 2) * nz
	if len(data) != data_length:
		raise Exception("Unexpected data length in file %s. Expected: %s ; Actual: %s" %(filename, str(data_length), str(len(data))))
	iterator = 0
	print "reshaping data in %s" %(dataset_name)
	out_grid = numpy.zeros(xx.shape)
	for k in range(nz):
		for j in range(0,ny+2):
			for i in range(0,nx+2):
				if j == j_axis_cut:
					out_grid[i,k] = data[iterator]
				iterator += 1
	print "max temp.: %s" %(str(numpy.max(out_grid)))
	print "plotting data in %s" %(dataset_name)
	pyplot.pcolor(zz, xx, out_grid, vmin=0, vmax=350)
	pyplot.autoscale(tight=True)
	# pyplot.xlabel("x at y=%i" %(j_axis_cut))
	# pyplot.ylabel("k")
	# pyplot.title(dataset_name)
	matplotlib.rc('font', **font)
	pyplot.colorbar() #.set_label("Temperature [K]")
	pyplot.savefig("./plot/%s.png" %(dataset_name))
	pyplot.close()