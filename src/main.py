#!/usr/bin/python

from SPNN import SPNN

if __name__ == '__main__':

	spnn = SPNN()
	#spnn.ensemble()
	#spnn.statEnsembleMultiProc()
	spnn.statEnsemble()

