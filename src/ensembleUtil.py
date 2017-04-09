

class ensembleUtil:

	def GNBPredict(self, x_train, y_train_class, x_test):
		from sklearn.naive_bayes import GaussianNB
		GNBModel= GaussianNB()
		GNBModel.fit(x_train, y_train_class)
		pgnb= GNBModel.predict(x_test)
		gnbprob = GNBModel.predict_proba(x_test)
		gnbprob = gnbprob*2 -1

		return pgnb
	
	def BNBPredict(self, x_train, y_train_class, x_test):

		from sklearn.naive_bayes import BernoulliNB
		BNBModel= BernoulliNB()
		BNBModel.fit(x_train, y_train_class)
		pbnb= BNBModel.predict(x_test)

		bnbprob = BNBModel.predict_proba(x_test)
		bnbprob = bnbprob*2 -1
			
		return pbnb


	def SGDPredict(self, x_train, y_train_class, x_test):

		from sklearn.linear_model import SGDClassifier
		sgdModel = SGDClassifier(loss="hinge", penalty="l2")
		sgdModel.fit(x_train, y_train_class)
		psgd = sgdModel.predict(x_test)
		return psgd


	def SVCPredict(self, x_train, y_train_class, x_test):
		from sklearn.svm import SVC
		SVCModel= SVC()
		SVCModel.fit(x_train, y_train_class)
		psvc=  SVCModel.predict(x_test)
		return psvc
		

	def LSVCPredict(self, x_train, y_train_class, x_test):
		from sklearn.svm import LinearSVC
		LSVCModel= LinearSVC()
		LSVCModel.fit(x_train, y_train_class)
		plsvc= LSVCModel.predict(x_test)
		return plsvc
		
	def SVRPredict(self, x_train, y_train, x_test):
		from sklearn import svm
		SVRModel = svm.SVR()
		SVRModel.fit(x_train, y_train)
		psvr = SVRModel.predict(x_test)
		return psvr

	def DTPredict(self, x_train, y_train_class, x_test):
		from sklearn import tree
		dtModel = tree.DecisionTreeClassifier()
		dtModel.fit(x_train, y_train_class)
		pdt = dtModel.predict(x_test)
		return pdt
