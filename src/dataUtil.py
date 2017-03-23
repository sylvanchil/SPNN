import pandas as pd
import numpy as np

from configure import Configure

class DataUtil:
	#load dataframe to numpy list
	def DataAndRate(self,df):
		data = df.as_matrix()
		rate = np.copy(data)
		for index in range(0, len(rate)-1):
			rate[index] = (rate[index]/rate[index+1][-1]-1)*10
		#discard first day data
		data = data[:-1]
		rate= rate[:-1]
		#flip array, data[0] = first day
		data= data[::-1]
		rate = rate[::-1]
		return data, rate
	
	def toMLPData(self, data, window,outputSize):
		dataList =[]
		for index in range(len(data) - (window+outputSize)+1 ):
			dataList.append(np.concatenate(data[index: index+ window+outputSize]))
		ndaRates = np.array(dataList)
		return ndaRates
	
	def toMLPTrainAndTest(self,inputData,testSize, outputSize):
		priceIndex = -1-4*(outputSize-1)
		priceIndexs = np.arange(priceIndex, 0, outputSize)

		trainSize = inputData.shape[0]-testSize
		trainData = inputData[:int(trainSize)]
		x_train = trainData[:,:Configure.window*4]
		#y_train = trainData[:,-1]
		y_train = trainData[:,priceIndexs]


		testData = inputData[int(trainSize):]
		x_test = testData[:,:Configure.window*4]
		#y_test = testData[:,-1]
		y_test = testData[:,priceIndexs]
		return x_train, y_train, x_test, y_test

