import numpy as np
from configure import Configure

class EvalUtil:
	def countHit(self,p, y_test):
		hit =0
		for index in range(p.shape[0]):
			if p[index] > 0 and y_test[index]>0 or p[index] < 0 and y_test[index] <0:
				hit = hit+1
		
		return hit, p.shape[0]

	def evalGain(self, gain):
		rate =np.copy( gain)
		for index in range(len(rate)-1, 0):
			rate[index] = rate[index]/rate[index-1]-1
		rate[0] = 0
		meanRate = np.mean(rate)
		dvarRate = np.std(rate)
		sharpeRatio = meanRate/dvarRate
		return sharpeRatio 



