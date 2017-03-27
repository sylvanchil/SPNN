from configure import Configure
from random import random, shuffle
import numpy as np

class SimulateTrading:
	def simWithSelection(self, predictions, y_tests):
		gain= []
		MoneyPool = 1000000
		predictions = np.array(predictions)
		y_tests = np.array(y_tests)

		for index in range(1,Configure.testSize):
			dayPred = predictions[:,index]
			dayY = y_tests[:,index]
			topTen = dayPred.argsort()[-10:]
	
			daygain=0
			
			for stock in topTen:
				if abs(dayY[stock]) <1:
					daygain =daygain + MoneyPool/10*dayPred[stock]*(dayY[stock]/10)
			#print daygain

			MoneyPool=MoneyPool+ daygain
			gain.append(MoneyPool)

		return gain



	def simWithNaiveMulti(self, p, y_test):
		predictions = []
		for index in range(len(p)):
			predictions.append(self.simWithNaive(p[index], y_test[index]))
		return predictions

	def simWithRandom(self, y_test):
		MoneyPool = 1000000
		randomSims =[]
		
		mu, sigma = 0.00, 0.40

		for index in range(100):
			
			MoneyPool = 1000000
			randomSim = []
			ranpred = np.random.normal(mu, sigma, len(y_test))
			
			for index2 in range(1, len(y_test)):
				if ranpred[index2]>0 and abs(y_test[index2]<1):
					MoneyPool = MoneyPool + MoneyPool*ranpred[index2]*(y_test[index2]/10)
				randomSim.append(MoneyPool)
			randomSims.append(randomSim)
		return np.array(randomSims)

	def simWithNaive(self, p , y_test):
		MoneyPool = 1000000
		predict = []
		
		for index in range(1,len(p)):
			if p[index]>0:
				if abs(y_test[index])<1:
					MoneyPool = MoneyPool + MoneyPool*p[index]*(y_test[index]/10)
			predict.append(MoneyPool)
		return np.array(predict)
