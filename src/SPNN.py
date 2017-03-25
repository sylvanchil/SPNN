import gc
import numpy as np
import io
from configure import Configure
from fileUtil import FileUtil
from dataUtil import DataUtil
from nnUtil import NNUtil
from visualize import VisualUtil
from simulateTrading import SimulateTrading
from evalUtil import EvalUtil
from random import random,shuffle

import pickle
import multiprocessing

class SPNN:
	
	fileUtil = FileUtil()
	dataUtil = DataUtil()
	nnUtil = NNUtil()
	vUtil = VisualUtil()
	evalUtil= EvalUtil()
	simUtil = SimulateTrading()

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
		
		model = self.nnUtil.buildLargeMLPModel(Configure.window*4, Configure.predictWindow)

		model = self.nnUtil.trainModel(model, x_train, y_train)
		model.save('%s/TM170325V1FOR%s'%(Configure.modelDirectory, Configure.stockCode))   	
		
		p=model.predict(x_test)
		
		p =p[:,0]
		y_test =y_test[:, 0]

		hit, total = self.evalUtil.countHit(p, y_test)
		print "Hit %s in %s , accuracy: %.4f" %(hit, total, hit*1.0000/total)
		gain = self.simUtil.simWithNaive(p, y_test)
		sharpeRatio = self.evalUtil.evalGain(gain)
		print "Gain sharpe ratio of : %.4f" % sharpeRatio
		self.vUtil.drawABGain(p,y_test, gain)

		return 

	'''
	2. multi input single model
		[RawData] -(FileUtil, DataUtil)-> [TrainXY, TestXY]
		[TranXY] -(DataUtil)-> TrainXY
		TrainXy-(nnUtil, MLP RNN LSTM)-> model
		model, [TestXy]->{prediction]
		[prediction] -(evaluaitonutil, countHit)-> [predict_evaluation] 
		[predict_evaluation] -(evaluationUtil, XXX) -> XXX
		
		[prediction] -(simutil, seleciton strategey)-> [gain]
		[prediction] -(simutil , naive strategy) -> [[gain]], [finalgain]
		
		[gain]-(EvaluationUtil, sharpe ratio, tstatic, stock index)-> sim_evaluation
		[gain]-(vUtil)->diagram
		
		[[gain]] -(vUtil, multiplot)-> diagram
		[[gain]] -(evaluationUtil, xxx) -> [XXX]

	3. multi inpu multi model
		[RawData] -(FileUtil, DataUtil)-> [TrainXY, TestXY]
		[TrainXY] -(nnUtil, MLP RNN, LSTM) -> [model]
		[model], [TestXY] -->[prediction]

		[prediction] -(evaluaitonutil, countHit)-> [predict_evaluation] 
		[predict_evaluation] -(evaluationUtil, XXX) -> XXX
		
		[prediction] -(simutil, seleciton strategey)-> [gain]
		[prediction] -(simutil , naive strategy) -> [[gain]], [finalgain]
		
		[gain]-(EvaluationUtil, sharpe ratio, tstatic, stock index)-> sim_evaluation
		[gain]-(vUtil)->diagram
		
		[[gain]] -(vUtil, multiplot)-> diagram
		[[gain]] -(evaluationUtil, xxx) -> [XXX]

	'''

	def MISM(self, 
			testSize = Configure.testSize,
			window= Configure.window,
			predictWindow= Configure.predictWindow):

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
				#if len(AllTrainList) >= 1:
				#	break
			AllTrainSet = np.concatenate([line for line in AllTrainList], axis= 0)
		np.random.shuffle(AllTrainSet)
		SaveData = False
		if SaveData:
			np.save('%s/TrainSet.npy'%Configure.midFileDirectory, AllTrainSet)
			for index in range(len(AllTestList) ):
				np.save('%s/TestDataSet/%s.npy'%(Configure.midFileDirectory, index), AllTestList[index])
		x_train, y_train= self.dataUtil.toXAndY(AllTrainSet, Configure.predictWindow)
		model = self.nnUtil.buildLargeMLPModel(Configure.window*4, Configure.predictWindow)
		model = self.nnUtil.trainModel(model, x_train, y_train)
		model.save('%s/TM170325V1'% Configure.modelDirectory)   	
		
		count= 0	
		for test in AllTestList:
			count = count +1
			x_test, y_test = self.dataUtil.toXAndY(test, Configure.predictWindow )
			p=model.predict(x_test)
			p =p[:,0]
			y_test =y_test[:, 0]
		
			hit, total = self.evalUtil.countHit(p, y_test)
			print "Hit %s in %s , accuracy: %.4f" %(hit, total, hit*1.0000/total)
			gain = self.simUtil.simWithNaive(p, y_test)
			sharpeRatio = self.evalUtil.evalGain(gain)
			print "Gain sharpe ratio of : %.4f" % sharpeRatio
			self.vUtil.drawABGain(p,y_test, gain)

			if count >0:
				break























