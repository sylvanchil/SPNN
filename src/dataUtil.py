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

	def toLSTMData(self, data, window, outputSize):
		dataList= []
		for index in range(len(data)-(window+outputSize)+1):
			dataList.append( data[index: index+window+outputSize])
		ndadata = np.array(dataList)
		return ndadata
	
	def toLSTMTrainAndTestSet(self, inputData, testSize):
		trainSize = inputData.shape[0]-testSize
		trainData = inputData[:int(trainSize)]
		testData = inputData[int(trainSize):]
		return trainData, testData

	def toLSTMXAndY(self, data, outputSize):
		x = data[:, :Configure.window]
		y = data[:, -outputSize:, -1]
		return x,y

	def toMLPData(self, data, window,outputSize):
		dataList =[]
		for index in range(len(data) - (window+outputSize)+1 ):
			dataList.append(np.concatenate(data[index: index+ window+outputSize]))
		ndaRates = np.array(dataList)
		return ndaRates

	def toMLPTrainAndTestSet(self, inputData, testSize):
		trainSize = inputData.shape[0]-testSize
		trainData = inputData[:int(trainSize)]
		testData = inputData[int(trainSize):]
		
		return trainData, testData

	def toXAndY(self, data, outputSize):
		priceIndex = -1-4*(outputSize-1)
		priceIndexs = np.arange(priceIndex, 0, 4)

		x = data[:,:Configure.window*4]
		y = data[:,priceIndexs]
		return x,y


