import gc
import numpy as np
import io
from configure import Configure
from fileUtil import FileUtil
from dataUtil import DataUtil
from nnUtil import NNUtil
from visualize import VisualUtil
from ensembleUtil import ensembleUtil
from simulateTrading import SimulateTrading
from evalUtil import EvalUtil
from random import random,shuffle
from sklearn.metrics import accuracy_score
import pickle
import multiprocessing

class SPNN:
	
	fileUtil = FileUtil()
	dataUtil = DataUtil()
	nnUtil = NNUtil()
	vUtil = VisualUtil()
	evalUtil= EvalUtil()
	simUtil = SimulateTrading()
	embUtil = ensembleUtil()

	def ensembleWorker(self, return_list, filename):
		print filename
		return_list.append(self.ensemble(stock=filename))
		return 

	def ensembleWorker2(self, return_list, fileQueue):
		while not fileQueue.empty():
			filename = fileQueue.get()
			print filename
			return_list.append(self.ensemble(stock=filename))
		return
		
	def statEnsembleMultiProc2(self, 
		testSize = Configure.testSize,
		window= Configure.window,
		predictWindow= Configure.predictWindow):
		
		fileList = [line for line in open(Configure.fileList)]
		shuffle(fileList)
		manager = multiprocessing.Manager()
		return_list = manager.list()
		fileQueue = manager.Queue()

		sample = True
		sampleSize= 100
		count =0
		for filename in fileList:
			count = count+1
			filename = filename.rstrip()
			fileQueue.put(filename)
			if sample and count >sampleSize:
				break

		
		jobs = []
	
		ncpus = multiprocessing.cpu_count()

		for i in range(ncpus):
			filename = filename.rstrip()
			p = multiprocessing.Process(target = self.ensembleWorker2, args=(return_list, fileQueue))
			jobs.append(p)
			p.start()

		for proc in jobs:
			proc.join()
		#Join will block the current process until the process we started finishes

		gains = np.array(return_list)
		print sum(gains)/len(gains)
		
		greater =0
		less =0
		remain = 0
		for i in gains:
			if i >1000000:
				greater = greater +1
			elif i ==1000000:
				remain = remain+1
			else:
				less = less +1

		print greater
		print less
		print remain


		self.vUtil.drawGain(gains)
		return


	def statEnsembleMultiProc(self, 
		testSize = Configure.testSize,
		window= Configure.window,
		predictWindow= Configure.predictWindow):
		
		fileList = [line for line in open(Configure.fileList)]
		shuffle(fileList)
		manager = multiprocessing.Manager()
		return_list = manager.list()
		
		count =0
		sample = True
		
		jobs = []
		
		for filename in fileList:
			count = count +1
			filename = filename.rstrip()
			p = multiprocessing.Process(target = self.ensembleWorker, args=(return_list, filename))
			jobs.append(p)
			p.start()
			if sample and count >=1000:
				break	

		for proc in jobs:
			proc.join()
		#Join will block the current process until the process we started finishes

		gains = np.array(return_list)
		print sum(gains)/len(gains)
		
		greater =0
		less =0
		remain = 0
		for i in gains:
			if i >1000000:
				greater = greater +1
			elif i ==1000000:
				remain = remain+1
			else:
				less = less +1

		print greater
		print less
		print remain


		self.vUtil.drawGain(gains)
		return

	def statEnsemble(self, 
		testSize = Configure.testSize,
		window= Configure.window,
		predictWindow= Configure.predictWindow):
		fileList = [line for line in open(Configure.fileList)]
		shuffle(fileList)
		gains = []	
		count =0
		sample = True
		for filename in fileList:
			count = count +1
			filename = filename.rstrip()
			print filename
			gain = self.ensemble(stock=filename)	
			if sample and count >=1000:
				break	
			gains.append(gain)
		print sum(gains)/len(gains)
		gains = np.array(gains)
		greater =0
		less =0
		remain = 0
		for i in gains:
			if i >1000000:
				greater = greater +1
			elif i ==1000000:
				remain = remain+1
			else:
				less = less +1

		print greater
		print less
		print remain

		self.vUtil.drawGain(gains)

		return

	def ensemble(self,
		stock= Configure.stock,
		testSize = Configure.testSize,
		window= Configure.window,
		predictWindow= Configure.predictWindow):

		df = self.fileUtil.csvToDataFrame(stock, window)
		if df.shape[0]-window+1 < Configure.testSize:
			print "Not enough data"
			return 1000000
		data, rate = self.dataUtil.DataAndRate(df)
		inputRate = self.dataUtil.toMLPData(rate, window, predictWindow)
		trainSet, testSet = self.dataUtil.toMLPTrainAndTestSet(inputRate, testSize)
		x_train, y_train= self.dataUtil.toXAndY( trainSet, Configure.predictWindow)
		x_test, y_test = self.dataUtil.toXAndY(testSet, Configure.predictWindow )

		y_train_class=np.copy(y_train)
		y_test_class= np.copy(y_test)

		y_train_class,flip = self.dataUtil.toClassLabel(y_train_class)
		if flip <2:
			return 1000000
		y_test_class,flip = self.dataUtil.toClassLabel(y_test_class)

		y_train_class= y_train_class[:,0]
		y_test_class = y_test_class[:,0]
		y_test = y_test[:,0]
		y_train = y_train[:,0]

		psvc = self.embUtil.SVRPredict(x_train, y_train, x_test)
		gain = self.simUtil.simWithNaive(psvc, y_test)
		if len(gain)==0:
			return 1000000
		return gain[-1]


	def SISMLSTM(self, 
			stock= Configure.stock,
			testSize = Configure.testSize,
			window= Configure.window,
			predictWindow= Configure.predictWindow):
		
		df = self.fileUtil.csvToDataFrame(stock, window)
		if df.shape[0]-window+1 < Configure.testSize:
			print "Not enough data"
			return
		data, rate = self.dataUtil.DataAndRate(df)

		inputRate= self.dataUtil.toLSTMData(rate, window, predictWindow)
		trainSet, testSet= self.dataUtil.toLSTMTrainAndTestSet(inputRate, testSize)
		x_train, y_train= self.dataUtil.toLSTMXAndY( trainSet, Configure.predictWindow)
		
		x_test, y_test = self.dataUtil.toLSTMXAndY(testSet, Configure.predictWindow )
		
		model = self.nnUtil.buildLSTMModel([Configure.window,4], Configure.predictWindow)
		model = self.nnUtil.trainModel(model, x_train, y_train)
		p=model.predict(x_test)
		p =p[:,0]
		y_test =y_test[:, 0]
		hit, total = self.evalUtil.countHit(p, y_test)
		print "Hit %s in %s , accuracy: %.4f" %(hit, total, hit*1.0000/total)
		gain = self.simUtil.simWithNaive(p, y_test)
		randomSims = self.simUtil.simWithRandom(y_test)
		self.vUtil.drawABGainRandom(p, y_test, gain, randomSims)
		return 
	
	def SISM(self, 
			stock= Configure.stock,
			testSize = Configure.testSize,
			window= Configure.window,
			predictWindow= Configure.predictWindow):
		
		df = self.fileUtil.csvToDataFrame(stock, window)
		if df.shape[0]-window+1 < Configure.testSize:
			print "Not enough data"
			return
		data, rate = self.dataUtil.DataAndRate(df)
		inputRate = self.dataUtil.toMLPData(rate, window, predictWindow)
		trainSet, testSet = self.dataUtil.toMLPTrainAndTestSet(inputRate, testSize)
		x_train, y_train= self.dataUtil.toXAndY( trainSet, Configure.predictWindow)
		
		x_test, y_test = self.dataUtil.toXAndY(testSet, Configure.predictWindow )
		
		model = self.nnUtil.buildMiniMLPModel(Configure.window*4, Configure.predictWindow)

		model = self.nnUtil.trainModel(model, x_train, y_train)
		
		p=model.predict(x_test)
		p =p[:,0]
		y_test =y_test[:, 0]

		hit, total = self.evalUtil.countHit(p, y_test)
		print "Hit %s in %s , accuracy: %.4f" %(hit, total, hit*1.0000/total)
		gain = self.simUtil.simWithNaive(p, y_test)
		randomSims = self.simUtil.simWithRandom(y_test)
		self.vUtil.drawABGainRandom(p, y_test, gain, randomSims)
		return 

	def MISM(self, 
			testSize = Configure.testSize,
			window= Configure.window,
			predictWindow= Configure.predictWindow):
		
		AllTrainSet = np.empty(0)
		AllTrainList= []
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
			total = 0
			for filename in fileList:
				total= total+1
				filename = filename.rstrip()
				df = self.fileUtil.csvToDataFrame(filename, Configure.window)
				
				if df.shape[0]-Configure.window+1 < Configure.testSize:
					print "skipped %s, not enough data" % filename
					continue
				
				data, rate = self.dataUtil.DataAndRate(df)
				inputRate = self.dataUtil.toMLPData(rate, Configure.window, Configure.predictWindow)
				trainSet, testSet = self.dataUtil.toMLPTrainAndTestSet(inputRate, Configure.testSize)
				AllTrainList.append(trainSet)
				AllTestList.append(testSet)
				gc.collect()
				print '%s: %s' %(total, filename)
				
				if len(AllTrainList) >= 100:
					break
			AllTrainSet = np.concatenate([line for line in AllTrainList], axis= 0)
		
		np.random.shuffle(AllTrainSet)
		
		SaveData = False
		if SaveData:
			np.save('%s/TrainSet.npy'%Configure.midFileDirectory, AllTrainSet)
			for index in range(len(AllTestList) ):
				np.save('%s/TestDataSet/%s.npy'%(Configure.midFileDirectory, index), AllTestList[index])
		
		x_train, y_train= self.dataUtil.toXAndY(AllTrainSet, Configure.predictWindow)
		model = self.nnUtil.buildMLPModel(Configure.window*4, Configure.predictWindow)
		model = self.nnUtil.trainModel(model, x_train, y_train)
	
		saveModel = True
		if saveModel:
			model.save('%s/TM170325V1'% Configure.modelDirectory)   	

		predictions =[]
		y_tests = []

		for test in AllTestList:
			x_test, y_test = self.dataUtil.toXAndY(test, Configure.predictWindow )
			p=model.predict(x_test)
			
			p =p[:,0]
			y_test =y_test[:, 0]
		
			predictions.append(p)
			y_tests.append(y_test)
	
		hits, totals, accuracies = self.evalUtil.countHits(predictions, y_tests)
		#self.vUtil.drawHist(accuracies)
		#naiveGains = self.simUtil.simWithNaiveMulti(predictions, y_tests)
		#self.vUtil.drawGains(naiveGains)
		gain = self.simUtil.simWithSelection(predictions,y_tests)

		self.vUtil.drawGain(gain)

	def MIMM(self, 
			testSize = Configure.testSize,
			window= Configure.window,
			predictWindow= Configure.predictWindow):
		
		fileList = [line for line in open(Configure.fileList)]
		shuffle(fileList)
		total = 0

		predictions = []
		y_tests = []
		accuracies = []
		gains= []

		for filename in fileList:
			
			filename = filename.rstrip()
			df = self.fileUtil.csvToDataFrame(filename, Configure.window)
			
			if df.shape[0]-Configure.window+1 < Configure.testSize:
				print "skipped %s, not enough data" % filename
				continue
	
			data, rate = self.dataUtil.DataAndRate(df)
			inputRate = self.dataUtil.toMLPData(rate, window, predictWindow)
			trainSet, testSet = self.dataUtil.toMLPTrainAndTestSet(inputRate, testSize)
			x_train, y_train= self.dataUtil.toXAndY( trainSet, Configure.predictWindow)
			
			x_test, y_test = self.dataUtil.toXAndY(testSet, Configure.predictWindow )
			
			model = self.nnUtil.buildMiniMLPModel(Configure.window*4, Configure.predictWindow)

			model = self.nnUtil.trainModel(model, x_train, y_train)
			
			p=model.predict(x_test)
			p =p[:,0]
			y_test =y_test[:, 0]


			hit, total = self.evalUtil.countHit(p, y_test)
			gain = self.simUtil.simWithNaive(p, y_test)

			predictions.append(p)
			y_tests.append(y_test)
			if total != 0:
				accuracies.append(hit*1.00/total)
			gains.append(gain)
			if len(predictions) >= 200:
				break	
		gain = self.simUtil.simWithSelection(predictions,y_tests)

		self.vUtil.drawGain(gain)
		#self.vUtil.drawGains(gains)
		return


