from multiprocessing.connection import Client
from multiprocessing.connection import Listener


workAddress = ('localhost', 6000)
resultAddress = ('localhost', 6001)
regexClient = Client(workAddress)
resultListener = Listener(resultAddress)
resultConnection = None

def getResult():
	global resultConnection
	if resultConnection == None:
		resultConnection = resultListener.accept()
	return resultConnection.recv()

regexClient.send({
	"regex": r'.*'
})
print str(getResult())
regexClient.send({
	"regex": r'.*',
	"text": "blub"
})
print str(getResult())
regexClient.send({
	"regex": r'(.*)',
	"text": "blub"
})
print str(getResult())
resultConnection.close()
regexClient.close()