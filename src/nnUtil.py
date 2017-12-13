
import numpy as np
from keras import optimizers
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.models import load_model
from configure import Configure


class NNUtil:
	def buildLSTMModel(self, dims, predictWindow):
		dropourRate = 0.5
		model = Sequential()
		model.add(LSTM(64, input_shape = (dims) ,return_sequences=True  ) )
		model.add(Dropout(dropourRate))
		model.add(LSTM(64, input_shape= (dims) , return_sequences=False  ) )
		model.add(Dropout(dropourRate))
		model.add(Dense(64, init= 'uniform', activation= 'relu'))
		model.add(Dropout(dropourRate))
		model.add(Dense(predictWindow, init= 'uniform', activation= 'linear'))
		model.compile(loss= 'mse', optimizer='adam', metrics= ['accuracy'])
		return model

	def buildMiniMLPModel(self, dims, predictWindow):
		dropourRate = 0.5
		
		model = Sequential()
		model.add(Dense(200, input_shape=(dims,)) )
		model.add(Dropout(dropourRate))
		model.add(Dense(256, init= 'uniform', activation= 'relu'))
		model.add(Dropout(dropourRate))
		model.add(Dense(predictWindow, init= 'uniform', activation= 'linear'))
		model.compile(loss= 'mse', optimizer='adam', metrics= ['accuracy'])
		return model

	def buildMLPModel(self, dims, predictWindow):
		dropourRate = 0.5
		
		sgd = optimizers.SGD(lr=0.1, decay=1e-3, momentum=0.9, nesterov=True)
		RMSprop = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)		
		model = Sequential()
		model.add(Dense(2048, input_shape=(dims,)) )
		model.add(Dropout(dropourRate))
		model.add(Dense(1024, init= 'uniform', activation= 'tanh'))
		model.add(Dropout(dropourRate))
		model.add(Dense(256, init= 'uniform', activation= 'relu'))
		model.add(Dropout(dropourRate))
		model.add(Dense(predictWindow, init= 'uniform', activation= 'linear'))
		model.compile(loss= 'mse', optimizer= RMSprop , metrics= ['accuracy'])
		#model.compile(loss= 'mse', optimizer='adam', metrics= ['accuracy'])
		return model

	def loadModel(self,model):
		return load_model(model)

	def trainModel(self,model, x_train, y_train):
		#model = self.nnUtil.buildLSTMModel([3, Configure.window, 1])

		model.fit(
			x_train,
			y_train,
			batch_size=8192,
			nb_epoch=Configure.epoch,
			validation_split=0.1,
			verbose=1
		)
		return model	
		


