import re
import sys
from optparse import OptionParser

def filterExceptions(exceptions, paths):
  exceptionsPiped = '|'.join([re.escape(exception) for exception in exceptions])
  pattern = re.compile(r'.*?(^|\W)+' + r'(' + exceptionsPiped + r')' + r'($|\W)+.*')
  return [path for path in paths if not pattern.match(path)]

##################### MAIN ##############################
#get all program arguments
parser = OptionParser()
parser.add_option("-e", "--exceptions", dest="exceptions",
                  help="exceptions to filter out from paths (space separated)")
parser.add_option("-p", "--paths", dest="paths",
                  help="paths to filter (space separated)")
parser.add_option("-d", "--debug", action="store_true", dest="debug",
                  help="show debug print in standard error output")
(options, args) = parser.parse_args()

if (not options.paths or options.paths.strip() == ''):
  print ''
  sys.exit(0)

exceptions = []
if (options.exceptions):
  exceptions = options.exceptions.strip().split(' ')
  exceptions = [exception.strip() for exception in exceptions if exception.strip() != ""]

paths = options.paths.strip().split(' ')
paths = [path.strip() for path in paths if path.strip() != ""]

try:
  if len(exceptions) > 0:
    paths = filterExceptions(exceptions, paths)
  print ' '.join(paths)
  sys.exit(0)
except Exception, e:
  sys.stderr.write('Error when checking whether %s is contained in %s: %s\n%s\n' %(options.name, options.path, str(e), traceback.format_exc()))
  sys.exit(64)