
import numpy as np
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.models import load_model

from configure import Configure


class NNUtil:
	def buildLSTMModel(self, dims):
		dropourRate = 0.5
		model = Sequential()
		model.add(LSTM(128, input_shape = (dims) ,return_sequences=False  ) )
		model.add(Dropout(dropourRate))
		model.add(Dense(64, init= 'uniform', activation= 'relu'))
		model.add(Dropout(dropourRate))
		model.add(Dense(1, init= 'uniform', activation= 'linear'))
		model.compile(loss= 'mse', optimizer='adam', metrics= ['accuracy'])
		return model

	def buildSimpleMLPModel(self, dims, predictWindow):
		dropourRate = 0.5
		model = Sequential()
		model.add(Dense(100, input_shape=(dims,)) )
		model.add(Dropout(dropourRate))
		model.add(Dense(500, init= 'uniform', activation= 'relu'))
		model.add(Dropout(dropourRate))
		model.add(Dense(predictWindow, init= 'uniform', activation= 'linear'))
		model.compile(loss= 'mse', optimizer='adam', metrics= ['accuracy'])
		return model

	def buildLargeMLPModel(self, dims, predictWindow):
		dropourRate = 0.5
		model = Sequential()
		model.add(Dense(1024, input_shape=(dims,)) )
		model.add(Dropout(dropourRate))
		model.add(Dense(1024, init= 'uniform', activation= 'relu'))
		model.add(Dropout(dropourRate))
		model.add(Dense(256, init= 'uniform', activation= 'relu'))
		model.add(Dropout(dropourRate))
		model.add(Dense(predictWindow, init= 'uniform', activation= 'linear'))
		model.compile(loss= 'mse', optimizer='adam', metrics= ['accuracy'])
		return model

	def loadModel(self,model):
		return load_model(model)

	def trainModel(self,model, x_train, y_train):
		#model = self.nnUtil.buildLSTMModel([3, Configure.window, 1])

		model.fit(
			x_train,
			y_train,
			batch_size=4096,
			nb_epoch=Configure.epoch,
			validation_split=0.1,
			verbose=1
		)
		return model	
		


