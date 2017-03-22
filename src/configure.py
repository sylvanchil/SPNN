class Configure:
	dataWarehouse= '/home/cj/Documents/Data/SPNN'
	dataFolder = '%s/data20170313' % dataWarehouse
	fileList = '%s/filelist' % dataWarehouse
	outputDirectory= '%s/output' % dataWarehouse
	modelDirectory= '%s/models' % outputDirectory

	trainedModel = '/home/cj/Documents/SPNN/src/models/trained.h5'
	stockName = '600894.ss'
	window = 60
	epoch = 500
	lever = 10000
	moneyPool = 1000000
	stockPool = 10000


