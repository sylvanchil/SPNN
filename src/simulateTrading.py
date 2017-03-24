from configure import Configure
import numpy as np

class SimulateTrading:
	

	def simulate3(self, p, y_test):
		MoneyPool= 100000

		predict = []

		for index in range(1,len(p)):
			
			if p[index]>0:

				MoneyPool = MoneyPool + MoneyPool*10*p[index]*(y_test[index]/10)
			predict.append(MoneyPool)
			print MoneyPool
		
		return np.array(predict)

	def simulate(self, p, y_test, yprice):
		MoneyPool= 0.0
		MoneyCheat = 0.0
		p = p.reshape(len(p))
		y_test=y_test.reshape(len(p))
		yprice= yprice.reshape(len(p))

		predict = []

		for index in range(1,len(p)):
			MoneyPool =MoneyPool+ p[index]*Configure.lever* ((yprice[index]-yprice[index-1])/yprice[0])
			predict.append(MoneyPool)
			MoneyCheat = MoneyCheat +y_test[index]* Configure.lever*((yprice[index]- yprice[index-1])/yprice[0])
		
		print MoneyPool
		print MoneyCheat
		return predict

	def simulate2(self, p, y_test, yprice):
		MoneyPool= 1000000
		MoneyCheat = 0.0
		p = p.reshape(len(p))
		y_test=y_test.reshape(len(p))
		yprice= yprice.reshape(len(p))

		predict = []
		
		for index in range(1,len(p)):
			MoneyPool =MoneyPool+ MoneyPool/1000*p[index]*Configure.lever* ((yprice[index]-yprice[index-1])/yprice[0])
			predict.append(MoneyPool)
			MoneyCheat = MoneyCheat +y_test[index]* Configure.lever*((yprice[index]- yprice[index-1])/yprice[0])
		
		print MoneyPool
		print MoneyCheat
		for index in range(len(predict)):
			predict[index] = predict[index]/1000000 -1 
		
		return predict

