import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class VisualUtil:

	def drawABGain(self, p,y_test, gain):
		plt.plot(p, color = 'red', label = 'prediction')
		plt.plot(y_test, color = 'blue', label = 'y_test')
		plt.legend(loc = 'upper left')
		plt.grid(True, which= 'both')
		plt.subplots_adjust(left=0.04, right=0.96, top=0.96, bottom=0.04)
		
		plt.figure()
		plt.plot(gain, color = 'grey')
		plt.grid(True, which= 'both')
		plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
		plt.show()

		return 

	def drawGain(self, gain):
		plt.plot(gain, color = 'grey')
		plt.grid(True, which= 'both')
		plt.subplots_adjust(left=0.04, right=0.96, top=0.96, bottom=0.04)
		plt.show()
		return 

	def drawAB(self, p, y_test):
		plt.plot(p, color = 'red', label = 'prediction')
		plt.plot(y_test, color = 'blue', label = 'y_test')
		plt.legend(loc = 'upper left')
		plt.grid(True, which= 'both')
		plt.subplots_adjust(left=0.04, right=0.96, top=0.96, bottom=0.04)
		plt.show()
		return 
	
	def drawReturn(self , p):
		parray = np.array(p)
		parray = parray[0,0]

		plt.plot(parray, color = 'grey')

		plt.legend(loc = 'upper left')
	
		plt.grid(True, which= 'both')

		plt.show()
		return

	def drawReturns(self, p):
		data = np.array(p)	
		
		mean = np.mean(data, axis=0)
	
		pdmean = pd.DataFrame(mean[1:])

		pdpc = pdmean.pct_change()


		sharpe = np.sqrt(504) * pdpc.mean() / pdpc.std()
		
		for sim in p:
			plt.plot(sim, color = 'grey')

		plt.plot(mean, color = 'blue', label = 'sharpe ratio: %.2f'% sharpe)

		plt.legend(loc = 'upper left')
	
		plt.grid(True, which= 'both')

		plt.show()
		return


	def drawPY(self, p, y_test):
		plt.plot(p, color = 'red', label = 'prediction')
		plt.plot(y_test, color = 'blue', label = 'y_test')
		plt.legend(loc = 'upper left')
		plt.grid(True, which= 'both')
		plt.subplots_adjust(left=0.04, right=0.96, top=0.96, bottom=0.04)
		plt.show()
		return 


	def vReturn(self, p):
		plt.plot(p, color = 'red', label = 'prediction')
		
		plt.legend(loc = 'upper left')
		plt.grid(True, which='both')

		plt.show()


		return


		
	def visulize(self, p, y_test):
		
		vp = p/10.0+1.0
		vy = y_test/10.0+1.0
		vp[0] = 1.0
		vy[0] = 1.0
		for index in range(1,len(p)):
			vp[index] = vp[index-1]*vp[index]
			vy[index] = vy[index-1]*vy[index]
		plt.plot(p, color = 'red', label = 'prediction')
		plt.plot(y_test, color= 'blue', label = 'Test')
		plt.legend(loc = 'upper left')
		plt.grid(True, which='both')
		
		plt.figure()
		plt.plot(vp, color = 'red', label = 'prediction')
		plt.plot(vy, color= 'blue', label = 'Test')
		plt.legend(loc = 'upper left')
		plt.grid(True, which='both')
		

		plt.show()


		return
	def visulize2(self, p, y_test):
		
		plt.plot(p, color = 'red', label = 'prediction')
		plt.plot(y_test, color= 'blue', label = 'Test')
		plt.legend(loc = 'upper left')
		plt.grid(True, which='both')

		plt.show()


		return


