import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

class VisualUtil:
	def drawGains(self, gains):
		meanOfGain= np.mean(gains, axis= 0)
		
		for gain in gains:
			plt.plot(gain, color = 'blue')
		plt.plot(meanOfGain, color = 'red')
		plt.grid(True, which= 'both')
		plt.show()

	def drawHist(self, data):
		bins = [0.2,0.8,0.01]
		plt.hist(data,bins=100)
		#plt.hist(data,bins=100)
		plt.show()
	def drawMultiTest(self, gains):
		font = { 
                'color':  'darkred',
                'weight': 'normal',
                'size': 16, 
                }   


		meanOfGain= np.mean(gains, axis= 0)

		for gain in gains:
			plt.plot(gain, color = 'grey')
		plt.plot(meanOfGain, color = 'blue')

		maxIndex =0
		minIndex =0
		maxPort =1000000
		minPort = 100000000


		for index in range(len(gains)):
			if gains[index][-1] > maxPort:
				maxPort= gains[index][-1]
				maxIndex = index
			if gains[index][-1] <minPort:
				minPort= gains[index][-1]
				minIndex = index

		plt.text(30, 1700000,'Max return: %.2f%%\nMin return: %.2f%%\nAvg return: %.2f%%' %(gains[maxIndex][-1]/10000-100, gains[minIndex][-1]/10000-100, meanOfGain[-1]/10000-100 ))

		plt.plot(gains[maxIndex], color = 'darkorange')
		plt.plot(gains[minIndex], color = 'brown')

		plt.grid(True, which = 'both')

		plt.title('2 Years Returns of %s Simulations'%(len(gains)), fontdict = font)
		plt.xlabel('Trading days', fontdict = font)
		plt.ylabel('Portfolio', fontdict = font)
		plt.show()


	
	def drawABGainRandom(self,p, y_test, gain, randomSims):
		plt.plot(p, color = 'red', label = 'prediction')
		plt.plot(y_test, color = 'blue', label = 'y_test')
		plt.legend(loc = 'upper left')
		plt.grid(True, which= 'both')
		plt.subplots_adjust(left=0.04, right=0.96, top=0.96, bottom=0.04)
		
		plt.figure()
		for sim in randomSims:
			plt.plot(sim, color = 'grey')

		meanOfRan= np.mean(randomSims, axis=0)

		plt.plot(gain, color = 'blue', label = 'gain')
		plt.plot(meanOfRan, color = 'red', label = 'mean of random')

		plt.grid(True, which= 'both')
		plt.subplots_adjust(left=0.04, right=0.96, top=0.96, bottom=0.04)
	
		plt.show()

		return 
		
	def drawGainAndRandom(self,gain, randomSims):
		for sim in randomSims:
			plt.plot(sim, color = 'grey')

		meanOfRan= np.mean(randomSims, axis=0)

		plt.plot(gain, color = 'blue', label = 'gain')
		plt.plot(meanOfRan, color = 'red', label = 'mean of random')

		plt.grid(True, which= 'both')
		plt.subplots_adjust(left=0.04, right=0.96, top=0.96, bottom=0.04)
		plt.show()
		return 

	
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

	def drawFile(self, filename):
		if os.path.isfile(filename) is not True:
			print 'file not exit'
		elif os.path.isdir(filename):
			#print all files
			pass
		else:
			pass
			#print one file

if __name__ == "__main__":
	vu = VisualUtil()


