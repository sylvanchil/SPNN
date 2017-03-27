import numpy as np
from configure import Configure

class EvalUtil:
	def countHits(self,p, y_test):
		hits = []
		totals = []
		accuracies = []

		for index in range(len(p)):
			hit, total=self.countHit(p[index], y_test[index])
			if total==0:
				continue
			accuracy = hit* (1.00)/total
			hits.append(hit)
			totals.append(total)
			accuracies.append(accuracy)

		return hits, totals, accuracies

	def countHit(self,p, y_test):
		hit =0
		total = 0
		for index in range(p.shape[0]):
			if y_test[index]==0:
				continue
			if p[index] > 0 and y_test[index]>0 or p[index] < 0 and y_test[index] <0:
				hit = hit+1
			total= total+1
		return hit, total

	#incorrect here
	def evalGain(self, gain):
		rate =np.copy( gain)
		for index in range(len(rate)-1, 0):
			rate[index] = rate[index]/rate[index-1]-1
			print rate[index]

		rate[0] = 0
		meanRate = np.mean(rate)
		dvarRate = np.std(rate)
		sharpeRatio = meanRate/dvarRate
		return sharpeRatio 



