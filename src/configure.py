class Configure:
	dataWarehouse= '/home/cj/Documents/Data/SPNN'
	dataFolder = '%s/data20170313' % dataWarehouse
	fileList = '%s/filelist' % dataWarehouse
	outputDirectory= '%s/output' % dataWarehouse
	modelDirectory= '%s/models' % outputDirectory
	midFileDirectory= '%s/midFile' % outputDirectory

	trainedModel = '/home/cj/Documents/SPNN/src/models/trained.h5'
	stockCode = '600894'
	marketCode = 'ss'
	stock = '%s.%s' % (stockCode, marketCode)
	# test 504 for 2 year
	testSize = 504
	window = 60
	predictWindow = 2
	#change back to 500
	epoch = 1000



