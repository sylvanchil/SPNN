import gc
import numpy as np
import io
from configure import Configure
from fileUtil import FileUtil
from dataUtil import DataUtil
from nnUtil import NNUtil
from visualize import VisualUtil
from simulateTrading import SimulateTrading
from random import random,shuffle
import pickle
import multiprocessing

class TestUtil:
	
	fileUtil = FileUtil()
	dataUtil = DataUtil()
	nnUtil = NNUtil()
	vUtil = VisualUtil()
	simUtil = SimulateTrading()


	def final(self):
		#predict all stock, get 2000* 252* 1 or 2 or 3or 4 array
		#sim 
		#	for 0 to 251
		# 		find top 10 from 2000
		
		AllTrainSet = np.empty(0)
		AllTestList= []
	
		useSavedFile = False 

		if useSavedFile:

			AllTrainSet = np.load('%s/TrainSet.npy'%Configure.midFileDirectory)

			AllTestList = []
		
			for index in range(2325):
				AllTestList.append(np.load('%s/TestDataSet/%s.npy'%(Configure.midFileDirectory, index)))

		else :

			fileList = [line for line in open(Configure.fileList)]
			shuffle(fileList)
			
			AllTrainList= []
			AllTestList = []

			total = 0
			for filename in fileList:
				total= total+1
				filename = filename.rstrip()
				df = self.fileUtil.csvToDataFrame(filename, Configure.window)
					
				#discard data with small size
				if df.shape[0]-Configure.window+1 < Configure.testSize:
					#print filename
					continue
				
				data, rate = self.dataUtil.DataAndRate(df)
				inputRate = self.dataUtil.toMLPData(rate, Configure.window, Configure.predictWindow)
				trainSet, testSet = self.dataUtil.toMLPTrainAndTestSet(inputRate, Configure.testSize)
				
				#AllTrainSet = np.concatenate([AllTrainSet, trainSet], axis = 0)
				
				AllTrainList.append(trainSet)

				AllTestList.append(testSet)
			
				gc.collect()

				print '%s: %s' %(total, filename)

				if len(AllTrainList) >= 1:
					break

			AllTrainSet = np.concatenate([line for line in AllTrainList], axis= 0)

		np.random.shuffle(AllTrainSet)

		SaveData = False
		if SaveData:
			np.save('%s/TrainSet.npy'%Configure.midFileDirectory, AllTrainSet)

			for index in range(len(AllTestList) ):
				np.save('%s/TestDataSet/%s.npy'%(Configure.midFileDirectory, index), AllTestList[index])
		
		x_train, y_train= self.dataUtil.toXAndY(AllTrainSet, Configure.predictWindow)
	
		model = self.nnUtil.buildCoreModel(Configure.window*4, Configure.predictWindow)
		
		model = self.nnUtil.trainModel(model, x_train, y_train)
		
		count= 0	
		for test in AllTestList:
			count = count +1
			x_test, y_test = self.dataUtil.toXAndY(test, Configure.predictWindow )
			p=model.predict(x_test)
			
			p =p[:,0]
			y_test =y_test[:, 0]

			self.vUtil.drawPY(p, y_test)

			if count >0:
				break


	def testMultiOutput(self):
		df = self.fileUtil.csvToDataFrame(Configure.stockName, Configure.window)
		data, rate= self.dataUtil.DataAndRate(df)
		inputRate = self.dataUtil.toMLPData(rate, Configure.window, Configure.predictWindow)
		
		x_train, y_train, x_test, y_test = self.dataUtil.toMLPTrainAndTest(inputRate, Configure.testSize, Configure.predictWindow)
	
		#trainData, testData= self.dataUtil.toMLPTrainAndTestSet(inputRate,Configure.testSize)
			
		#x_train, y_train = self.dataUtil.toXAndY(trainData, Configure.predictWindow)
		#x_test, y_test= self.dataUtil.toXAndY(testData, Configure.predictWindow)

		print x_train.shape
		print y_train.shape
		print x_test.shape
		print y_test.shape
		return


		model = self.nnUtil.buildSimpleMLPModel(Configure.window*4, Configure.predictWindow)
		
		model = self.nnUtil.trainModel(model, x_train, y_train)

		p = model.predict(x_test)

		p =p[:,0]
		y_test =y_test[:, 0]
		simReturnSet =[]
		
		simReturnSet.append(self.simUtil.simulate3(p,y_test))
		simReturnSet = np.array(simReturnSet)
		self.vUtil.drawReturns(simReturnSet)


	def testSampleAndTrain(self):
		df = self.fileUtil.csvToDataFrame(Configure.stockName, Configure.window)
		data, rate= self.dataUtil.DataAndRate(df)
		inputRate = self.dataUtil.toInputData(rate, Configure.window, 1)

		inputPrice = self.dataUtil.toInputData(data, Configure.window, 1)

		tmpx, tmpy, tmppricex, pricey = self.dataUtil.trainAndTestSet(inputPrice, 200)

		tmpx, tmpy, x_test, y_test = self.dataUtil.trainAndTestSet(inputRate, 200)

		data = data[:0]
		rate = rate[:0]
	
		print data.shape
		print rate.shape

		samples = [line for line in open(Configure.fileList) if random() < 0.1]
		for filename in samples:
			filename = filename.rstrip()
			if filename!= Configure.stockName:
				print filename
				df = self.fileUtil.csvToDataFrame(Configure.stockName, Configure.window)
				data2, rate2 = self.dataUtil.DataAndRate(df)
				data = np.concatenate((data,data2),axis=0)
				rate = np.concatenate((rate,rate2), axis=0)
			gc.collect()
	
		inputData = self.dataUtil.toInputData(rate, Configure.window, 1)
		np.random.shuffle(inputData)
		gc.collect()
		
		np.random.shuffle(inputData)

		x_train, y_train, tmpx, tmpy = self.dataUtil.trainAndTestSet(inputData, 2000)
	
		model = self.nnUtil.buildCoreModel(Configure.window*4)
		
		model = self.nnUtil.trainModel(model, x_train, y_train)

		model.save('./mar19_trainedv3.h5')

		p = model.predict(x_test)

		hit = 0
		for index in range(len(p)):
			if p[index]>0 and y_test[index] >0 or p[index]<=0 and y_test[index]<=0:
				hit = hit+1

		print hit
		print len(p)

		self.vUtil.visulize2(p, y_test)

		return 

	def testSingleSimAndVisual(self):
		stock= Configure.stockName
	
		model=self.nnUtil.loadModel('/home/cj/Documents/SPNN/data/output/mar19_trainedv3.h5')
		df = self.fileUtil.csvToDataFrame(stock, Configure.window)
		data, rate= self.dataUtil.DataAndRate(df)
		inputData = self.dataUtil.toInputData(rate, Configure.window, 1)
		inputPrice = self.dataUtil.toInputData(data, Configure.window, 1)

		xtmp, ytmp, xtmp2, yprice= self.dataUtil.trainAndTestSet(inputPrice, 200)

		x_train, y_train, x_test, y_test = self.dataUtil.trainAndTestSet(inputData, 200)
		p = model.predict(x_test)
		print stock
		simReturn = self.simUtil.simulate(p,y_test, yprice)
		
		self.vUtil.vReturn(simReturn)

	def testSampleTest(self):
		from sets import Set
		trainedSet = Set()

		with open('%s/log'% Configure.outputDirectory) as f:
			for filename in f:
				filename = filename.rstrip()
				trainedSet.add(filename)
	
		model=self.nnUtil.loadModel('%s/TM170319V3.h5'% Configure.modelDirectory)

		simReturnSet= []


		samples = [line for line in open(Configure.fileList)]
		shuffle(samples)	
	

		for filename in samples:
			filename = filename.rstrip()
			if(filename not in trainedSet):
			
				df = self.fileUtil.csvToDataFrame(filename, Configure.window)
				if df.size< Configure.window*10:
					continue 
				data, rate= self.dataUtil.DataAndRate(df)
				inputData = self.dataUtil.toInputData(rate, Configure.window, 1)
				inputPrice = self.dataUtil.toInputData(data, Configure.window, 1)

				xtmp, ytmp, xtmp2, yprice= self.dataUtil.trainAndTestSet(inputPrice, 504)

				x_train, y_train, x_test, y_test = self.dataUtil.trainAndTestSet(inputData, 504)
				p = model.predict(x_test)
			
				simReturnSet.append(self.simUtil.simulate(p,y_test, yprice))
			gc.collect()

			if len(simReturnSet)>200:
				break
	
		self.vUtil.drawReturns(simReturnSet)

	def testSimAll(self):
		model=self.nnUtil.loadModel('./mar19_trainedv3.h5')
		with open(Configure.fileList) as f:
			for filename in f:
				filename = filename.rstrip()
				self.testStockWithTrainedModel(filename, model)

	def testStockWithTrainedModel(self,stock, model):
		df = self.fileUtil.csvToDataFrame(stock, Configure.window)
		if df.size< Configure.window*4:
			return 
		data, rate= self.dataUtil.DataAndRate(df)
		inputData = self.dataUtil.toInputData(rate, Configure.window, 1)
		inputPrice = self.dataUtil.toInputData(data, Configure.window, 1)

		xtmp, ytmp, xtmp2, yprice= self.dataUtil.trainAndTestSet(inputPrice, 200)

		x_train, y_train, x_test, y_test = self.dataUtil.trainAndTestSet(inputData, 200)
		p = model.predict(x_test)
		print stock
		self.simUtil.simulate(p,y_test, yprice)
	
	def testDefault(self):
		stock= Configure.stockName
		model=self.nnUtil.loadModel('./mar19_trained.h5')
		self.testStockWithTrainedModelDisplay(stock,model)


	def testStockWithTrainedModelDisplay(self,stock, model):
		df = self.fileUtil.csvToDataFrame(stock, Configure.window)
		data, rate= self.dataUtil.DataAndRate(df)
		inputData = self.dataUtil.toInputData(rate, Configure.window, 1)
		inputPrice = self.dataUtil.toInputData(data, Configure.window, 1)

		xtmp, ytmp, xtmp2, yprice= self.dataUtil.trainAndTestSet(inputPrice, 200)

		x_train, y_train, x_test, y_test = self.dataUtil.trainAndTestSet(inputData, 200)
		p = model.predict(x_test)
		print stock
		self.simUtil.simulate(p,y_test, yprice)
		self.vUtil.visulize2(p, y_test)



	def testMulProc(self):

		def worker(i, j ):
			for index in range(10000000):
				print '%s of worker No.%s, %s' % (index, i, j)
			return

		jobs = []
		for i in range(4):
			for j in range(4):
				p = multiprocessing.Process(target=worker, args=(i, j))
				jobs.append(p)
				p.start()
			

	def testMakePickle(self):
		df = self.fileUtil.csvToDataFrame(Configure.stockName, Configure.window)
		data, rate= self.dataUtil.DataAndRate(df)
		with open(Configure.fileList) as f:
			for filename in f:
				filename = filename.rstrip()
				if filename!= Configure.stockName:
					print filename
					df = self.fileUtil.csvToDataFrame(Configure.stockName, Configure.window)
					data2, rate2 = self.dataUtil.DataAndRate(df)
					data = np.concatenate((data,data2),axis=0)
					rate = np.concatenate((rate,rate2), axis=0)
				gc.collect()
		
		with open('./data.pkl', 'wb') as f:
			pickle.dump(data, f)
	
		with open('./rate.pkl', 'wb') as f:
			pickle.dump(rate, f)
	
		


		#inputData = self.dataUtil.toInputData(rate, Configure.window, 1)

	def testWithPickle(self):

		with open('./rate.pkl','rb') as f:
		    rate = pickle.load(f)
		
		rate = rate[:-200]

		np.random.shuffle(rate)

		inputData = self.dataUtil.toInputData(rate, Configure.window,1)

	def test4(self):
		df = self.fileUtil.csvToDataFrame(Configure.stockName, Configure.window)
		data, rate= self.dataUtil.DataAndRate(df)
		with open(Configure.fileList) as f:
			for filename in f:
				filename = filename.rstrip()
				if filename!= Configure.stockName:
					print filename
					df = self.fileUtil.csvToDataFrame(Configure.stockName, Configure.window)
					data2, rate2 = self.dataUtil.DataAndRate(df)
					data = np.concatenate((data,data2),axis=0)
					rate = np.concatenate((rate,rate2), axis=0)
				gc.collect()
		


		inputData = self.dataUtil.toInputData(rate, Configure.window, 1)

#		with open('./inputData.pkl','rb') as f:
#		    inputData = pickle.load(f)

		x_train, y_train, x_test, y_test = self.dataUtil.trainAndTestSet(inputData, 2000)

		gc.collect()

		print ('finish test data')
		np.random.shuffle(inputData)

		#with open('./inputData.pkl', 'wb') as f:
		#	pickle.dump(inputData, f)
		


		x_train, y_train, tmpx, tmpy = self.dataUtil.trainAndTestSet(inputData, 2000)

		#inputData = self.dataUtil.toLSTMData(data, Configure.window, 1)
		#x_train, y_train, x_test, y_test = self.dataUtil.LSTMtrainAndTestSet(inputData, 200)

	
		model = self.nnUtil.buildCoreModel(Configure.window*4)

		model.save('./untrained.h5')

		#model = self.nnUtil.buildLSTMModel((Configure.window,4))
		model = self.nnUtil.trainModel(model, x_train, y_train)

		model.save('./trained.h5')

		p = model.predict(x_test)

		hit = 0
		for index in range(len(p)):
			if p[index]>0 and y_test[index] >0 or p[index]<=0 and y_test[index]<=0:
				hit = hit+1

		print hit
		print len(p)

		self.vUtil.visulize2(p, y_test)
	
	
	def test2(self):
		df = self.fileUtil.csvToDataFrame('600894.ss', 0)
		#data = self.dataUtil.loadRate(df[::-1],Configure.window)
		#x_train, y_train, x_test, y_test= self.dataUtil.loadData2(df[::-1], Configure.window)
		x_train, y_train, x_test, y_test= self.dataUtil.loadRate(df[::-1], Configure.window)
		#model = self.nnUtil.buildCoreModel([4, Configure.window, 1])
		model = self.nnUtil.buildLSTMModel([4, Configure.window, 1])
		model = self.nnUtil.trainModel(model, x_train, y_train)

		trainScore = model.evaluate(x_train, y_train, verbose=0)
		print('Train Score: %.2f MSE (%.2f RMSE)') % (trainScore[0], math.sqrt(trainScore[0]))	
		
		testScore = model.evaluate(x_test, y_test, verbose=0)

		print('Test Score: %.2f MSE (%.2f RMSE)') % (testScore[0], math.sqrt(testScore[0]))	
		
		p = model.predict(x_test)
	
		self.simUtil.simul(p, y_test)
		self.simUtil.naiveSimul(p, y_test)
		
		self.vUtil.visulize2(p, y_test)	


		

	def test(self, stockName):
		df = self.fileUtil.csvToDataFrame(stockName, 0)
	#	x_train, y_train, x_test, y_test= self.dataUtil.loadData(df[::-1], Configure.window)
		x_train, y_train, x_test, y_test= self.dataUtil.loadRate(df[::-1], Configure.window)

		model = self.nnUtil.buildLSTMModel([4, Configure.window, 1])
		#model = self.nnUtil.buildCoreModel([4, Configure.window, 1])
		model = self.nnUtil.trainModel(model, x_train, y_train)


		trainStock= ['002114.sz','000818.sz','000985.sz','601919.ss','600691.ss']

		trainScore = model.evaluate(x_train, y_train, verbose=0)
		print('Train Score: %.2f MSE (%.2f RMSE)') % (trainScore[0], math.sqrt(trainScore[0]))	
		
		testScore = model.evaluate(x_test, y_test, verbose=0)

		print('Test Score: %.2f MSE (%.2f RMSE)') % (testScore[0], math.sqrt(testScore[0]))	
		
		p = model.predict(x_test)
	
		total = len(p)
		hit =0

		for index in range (len(p)):
			if p[index]>0 and y_test[index]>0 or p[index]<=0 and y_test[index]<=0:
				hit = hit+1

		print (hit)
		print (total)
	
		self.simUtil.simul(p, y_test)
		self.simUtil.naiveSimul(p, y_test)
		
		self.vUtil.visulize2(p, y_test)	


		return 
	
#grave yard
	def test5(self):
		df = self.fileUtil.csvToDataFrame(Configure.stockName, Configure.window)
		data, rate= self.dataUtil.DataAndRate(df)
		inputData = self.dataUtil.toInputData(rate, Configure.window, 1)
		x_train, y_train, x_test, y_test = self.dataUtil.trainAndTestSet(inputData, 200)
		model = load_model('./trained.h5')

		p = model.predict(x_test)

		hit = 0
		for index in range(len(p)):
			if p[index]>0 and y_test[index] >0 or p[index]<=0 and y_test[index]<=0:
				hit = hit+1

		print hit
		print len(p)

		self.vUtil.visulize(p, y_test)

	
	def testAll(self):
		
		with open(Configure.fileList) as f:
			for filename in f:
				filename = filename.rstrip()
				self.test(filename)
		return


