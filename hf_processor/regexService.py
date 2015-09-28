from multiprocessing.connection import Client
from multiprocessing.connection import Listener
import re

class RegexService(object):
	patternsByRegex = None

	def __init__(self):
		self.patternsByRegex = {}
		self.matchgroupsByRegexAndText = {}

	def processMessage(self, message):
		regex = message.get('regex')
		result = {"error": None}
		if regex == None:
			result["error"] = "no regex in message - something is wrong with your client"
			return result
		text = message.get('text')
		pattern = self.patternsByRegex.get(regex)
		if pattern == None:
			print "compiling previously unseen regex: %s" %(regex)
			pattern = re.compile(regex, re.IGNORECASE)
			self.patternsByRegex[regex] = pattern
		if text == None:
			return result
		result["matchgroups"] = None
		matchgroups = self.matchgroupsByRegexAndText.get((regex, text))
		if matchgroups == None:
			match = pattern.match(text)
			if match != None:
				matchgroups = match.groups()
				self.matchgroupsByRegexAndText[(regex, text)] = matchgroups
		if matchgroups == None:
			result["error"] = "no match"
			return result
		result["matchgroups"] = matchgroups
		return result

workAddress = ('localhost', 6000)
resultAddress = ('localhost', 6001)
listener = Listener(workAddress)
service = RegexService()
patterns = {}
while True:
	connection = listener.accept()
	resultClient = Client(resultAddress)
	while True:
		try:
			message = connection.recv()
			resultClient.send(service.processMessage(message))
		except EOFError:
			resultClient.close()
			connection.close()
			break
listener.close()