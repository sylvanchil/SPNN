from configure import Configure
from random import random, shuffle
import numpy as np

class SimulateTrading:
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
