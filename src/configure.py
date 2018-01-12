class Configure:
	dataWarehouse= '/home/congq/data/data'
	dataFolder = '%s/data20170403' % dataWarehouse
	fileList = '%s/filelist' % dataWarehouse
	outputDirectory= '%s/output' % dataWarehouse
	modelDirectory= '%s/models' % outputDirectory
	midFileDirectory= '%s/midFile' % outputDirectory

	
	sampleSize = 40

	useSavedFile = False
	saveData = False
	saveModel = False

	drawMISMResult = True

	testRounds = 1

	trainedModel = '/home/cj/Documents/SPNN/src/models/trained.h5'
	stockCode = '600050'
	marketCode = 'ss'
	stock = '%s.%s' % (stockCode, marketCode)
	# test 504 for 2 year
	testSize = 504
	window = 55
	predictWindow = 2
	#change back to 400
	epoch = 10




