class Configure:
	dataWarehouse= '/home/cj/Documents/Data/SPNN'
	dataFolder = '%s/data20170313' % dataWarehouse
	fileList = '%s/filelist' % dataWarehouse
	outputDirectory= '%s/output' % dataWarehouse
	modelDirectory= '%s/models' % outputDirectory

	trainedModel = '/home/cj/Documents/SPNN/src/models/trained.h5'
	stockName = '002051.sz'
	testSize = 252
	window = 40
	predictWindow = 5
	#change back to 500
	epoch = 500


