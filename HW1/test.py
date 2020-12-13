import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
import pickle
import seaborn as sns
class model():
	def __init__(self, IN, h1, h2,OUT,learning_rate=1e-3, learning_rate_decay=1,steps=100,batch_size=4):
		self.IN = IN
		self.OUT = OUT
		np.random.seed(0)
		self.params = {}
		self.params['w1'] = np.random.randn(IN, h1) / np.sqrt(IN)
		self.params['b1'] = np.zeros(h1)
		self.params['w2'] = np.random.randn(h1, h2) / np.sqrt(h1)
		self.params['b2'] = np.zeros(h2)
		self.params['w3'] = np.random.randn(h2, OUT) / np.sqrt(h2)
		self.params['b3'] = np.zeros(OUT)
		self.lr = learning_rate
		self.decay = learning_rate_decay
		self.steps = steps
		self.batch_size = batch_size
	def val_loss(self,X,y):
		# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # validation loss 										 	  #
		# This function is on order to track the loss of testing data #
		# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
		out = self.forward(X)
		M = out - np.max(out)
		correct_f = M[np.arange(X.shape[0]), y]
		log_p = np.log(np.sum(np.exp(M), axis=1))
		loss = np.sum(-correct_f + log_p) / X.shape[0]
		return loss
	def loss(self, X, y):
		# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # foward propogation									 	  #
		# Compute the output of model by matrix multiplication	 	  #
		# Using np.maximum to implement ReLu layer 					  #
		# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
		h1 = np.maximum(X @ self.params['w1'] + self.params['b1'] , 0)
		h2 = np.maximum(h1 @ self.params['w2'] + self.params['b2'] , 0)
		out = h2 @ self.params['w3'] + self.params['b3']
		# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
		#	back propgation							     			  #
		#	The goal here is to using the chian rule to find out the  #
		#	value of gradient and bias in order to update the model   #
		# 	delta is the gradient of this mini batch				  # 
		#	Cross_loss is the loss compute by cross entropy			  #
		#	Computation is done layer by layer. using the error that  #
		# 	has computed in upper layer and then multiply with 	the	  #
		# 	weight of current layer. After that multiple it with the  #
		#	input of current layer									  #
		# 	the bias is just a simplt summation of the loss	of upper  #
		#	layer.													  #
		# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
		M = out - np.max(out)
		correct_f = M[np.arange(X.shape[0]), y]
		log_p = np.log(np.sum(np.exp(M), axis=1))
		loss = np.sum(-correct_f + log_p) / X.shape[0]
		delta = {} 
		prob = np.exp(M) / np.sum(np.exp(M), axis=1, keepdims=True)
		cross_loss = prob
		cross_loss[range(X.shape[0]), y] -= 1
		cross_loss /= X.shape[0]
		# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
		#	compute weight											  #
		# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
		delta['w3'] = h2.T @ cross_loss
		dh2 = cross_loss @ self.params['w3'].T
		dh2[h2 == 0] = 0
		delta['w2'] = h1.T @ dh2 
		dh1 = dh2 @ self.params['w2'].T
		dh1[h1 == 0] = 0
		delta['w1'] = X.T @ dh1
		# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
		#	compute bias											  #
		# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
		delta['b3'] = np.sum(cross_loss, axis=0)
		delta['b2'] = np.sum(dh2, axis=0)
		delta['b1'] = np.sum(dh1, axis=0)
		return loss, delta

	def train(self, X, y, X_val, y_val):
		# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
		#	train : train the model by gradient descend requires 	  #
		#	iteration.												  #
		#	We will handle input and output by loop	for a mini batch  #
		# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
		num_train = X.shape[0]
		number_epoch = max(num_train / self.batch_size, 1)
		val_acc = 0.0

		loss_ = []
		val_loss_ = []
		train_acc_history = []
		val_acc_history = []
		epoch_loss = 0
		for i in range(self.steps):
		# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
		# 	-Get the mini batch										  #
		# 	-compute gradient										  #
		# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #	
			batch_id = np.random.choice(a=num_train, size=self.batch_size, replace=True)
			X_batch = X[batch_id]
			y_batch = y[batch_id]

			loss, grads = self.loss(X_batch, y=y_batch)
			epoch_loss += loss
		# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
		#	update the parameter in model by SGD					  #
		#	Which is delta * learning rate							  #
		# 	the learning rate should be dacay (SGD)					  #
		# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
			for param in self.params:
				self.params[param] -= self.lr* grads[param]
						# Decay learning rate
			if i % (number_epoch*100) == 0:
				self.lr *= self.decay
			# if it > 400 and it % (iterations_per_epoch*40) == 0:
			# 	self.lr *= self.decay
			# if it > 100000 and it% (iterations_per_epoch * 50) == 0:
			# 	self.lr *= 0.85
		# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
		# 	tracking the performance of model						  #
		#	-print loss and accuracy								  #
		#	-append to list for plot the diagram					  #
		#	-save the best model									  #
		# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
			if i % 100 == 0:
				print('loss: %f ,val_acc: %f' % (loss, val_acc))
			history_high = 0.0
			if i % number_epoch == 0:
				loss_.append(epoch_loss/number_epoch)
				epoch_loss = 0
				val_loss_.append(self.val_loss(X_val,y_val))
				train_acc = (self.predict(X) == y).mean()
				val_acc = (self.predict(X_val) == y_val).mean()
				if val_acc > history_high:
					history_high = val_acc
					pickle.dump(self.params,open('best_modele3.p','wb')) 
				train_acc_history.append(1.0-train_acc)
				val_acc_history.append(1.0-val_acc)
		return {
		'loss_history': loss_,
		'val_loss_history': val_loss_,
		'train_history': train_acc_history,
		'val_history': val_acc_history,
		}

	def forward(self,X):
		# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # foward propogation - this code is for convenient when no    #
		# update is needed											  #
		# Compute the output of model by matrix multiplication	 	  #
		# Using np.maximum to implement ReLu layer 					  #
		# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
		scores = np.maximum(X@self.params['w1'] + self.params['b1'], 0) @ self.params['w2'] + self.params['b2']
		h1 = np.maximum(X @ self.params['w1'] + self.params['b1'], 0)
		h2 = np.maximum(h1 @ self.params['w2'] + self.params['b2'], 0)
		scores = h2 @ self.params['w3'] + self.params['b3']
		return scores

	def predict(self, X):
		return np.argmax(self.forward(X), axis=1)


def main():
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #   question 6 : load parameter and get output                #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
	steps = 100000
	lr = 0.099
	decay = 0.97
	batch = 8
	mymodel = model(8,3,3,2,lr,decay,steps,batch)
	bestparameter = pickle.load(open('best.p','rb'))
	for param in mymodel.params:
	    mymodel.params[param] = bestparameter[param] 
	Scalar = pickle.load(open('scaler.p','rb'))
	mydata = np.asarray([[0,5,1,0,500,0,0,1]])
	d_std = Scalar.transform(mydata)
	score = mymodel.forward(d_std)
	print(score)

if __name__ == "__main__":
    main()