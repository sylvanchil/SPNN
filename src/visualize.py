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
		if os.path.isfile(filename) not True:
			print 'file not exit'
		else if os.path.isdir(filename):
			#print all files

		else:
			#print one file

if __name__ == "__main__":
	vu = VisualUtil()


