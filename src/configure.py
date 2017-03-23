class Configure:
	dataWarehouse= '/home/cj/Documents/Data/SPNN'
	dataFolder = '%s/data20170313' % dataWarehouse
	fileList = '%s/filelist' % dataWarehouse
	outputDirectory= '%s/output' % dataWarehouse
	modelDirectory= '%s/models' % outputDirectory

	trainedModel = '/home/cj/Documents/SPNN/src/models/trained.h5'
	stockName = '600894.ss'
	testSize = 200
	window = 40
	predictWindow = 2
	#change back to 500
	epoch = 200
	lever = 10000
	moneyPool = 1000000
	stockPool = 10000


