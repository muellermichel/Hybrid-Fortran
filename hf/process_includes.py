import re, fileinput, os, sys

if len(sys.argv) != 2:
	raise Exception("process_includes needs to be called with file- or directory path")
directory_path = None
file_path = None
if os.path.isdir(sys.argv[1]):
	directory_path = sys.argv[1]
else:
	directory_path = os.path.dirname(sys.argv[1])
	file_path = sys.argv[1]

includePattern = re.compile(r'\s*include\s+[\'"]?([\w\.\\\/]*)[\'"]?\s*', re.IGNORECASE)

fileInputObject = None
if file_path:
	fileInputObject = fileinput.input(file_path)
else:
	fileInputObject = fileinput.input("-")

for line in fileInputObject:
	includeMatch = includePattern.match(line)
	if not includeMatch:
		print line
		continue
	includePath = includeMatch.group(1)
	includeFile = open(os.path.join(directory_path, includePath),'r')
	print includeFile.read()
	includeFile.close()