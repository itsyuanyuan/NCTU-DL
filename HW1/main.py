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
        	# validation loss 		                              #
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
        	# foward propogation    				      #
		# Compute the output of model by matrix multiplication	      #
		# Using np.maximum to implement ReLu layer        	      #
		# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
		h1 = np.maximum(X @ self.params['w1'] + self.params['b1'] , 0)
		h2 = np.maximum(h1 @ self.params['w2'] + self.params['b2'] , 0)
		out = h2 @ self.params['w3'] + self.params['b3']
		# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
		#	back propgation					          #
		#	The goal here is to using the chian rule to find out the  #
		#	value of gradient and bias in order to update the model   #
		# 	delta is the gradient of this mini batch		  # 
		#	Cross_loss is the loss compute by cross entropy		  #
		#	Computation is done layer by layer. using the error that  #
		# 	has computed in upper layer and then multiply with the	  #
		# 	weight of current layer. After that multiple it with the  #
		#	input of current layer					  #
		# 	the bias is just a simplt summation of the loss	of upper  #
		#	layer.							  #
		# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
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
		#	compute weight					      #
		# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
		delta['w3'] = h2.T @ cross_loss
		dh2 = cross_loss @ self.params['w3'].T
		dh2[h2 == 0] = 0
		delta['w2'] = h1.T @ dh2 
		dh1 = dh2 @ self.params['w2'].T
		dh1[h1 == 0] = 0
		delta['w1'] = X.T @ dh1
		# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
		#	compute bias			    		      #
		# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
		delta['b3'] = np.sum(cross_loss, axis=0)
		delta['b2'] = np.sum(dh2, axis=0)
		delta['b1'] = np.sum(dh1, axis=0)
		return loss, delta

	def train(self, X, y, X_val, y_val):
		# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
		#	train : train the model by gradient descend requires 	  #
		#	iteration.						  #
		#	We will handle input and output by loop	for a mini batch  #
		# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
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
		# 	-Get the mini batch				      #
		# 	-compute gradient				      #
		# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #	
			batch_id = np.random.choice(a=num_train, size=self.batch_size, replace=True)
			X_batch = X[batch_id]
			y_batch = y[batch_id]

			loss, grads = self.loss(X_batch, y=y_batch)
			epoch_loss += loss
		# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
		#	update the parameter in model by SGD		      #
		#	Which is delta * learning rate			      #
		# 	*the learning rate would dacay  		      #
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
		# 	tracking the performance of model    		      #
		#	-print loss and accuracy			      #
		#	-append to list for plot the diagram		      #
		#	-save the best model				      #
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
		# update is needed					      #
		# Compute the output of model by matrix multiplication	      #
		# Using np.maximum to implement ReLu layer 		      #
		# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
		scores = np.maximum(X@self.params['w1'] + self.params['b1'], 0) @ self.params['w2'] + self.params['b2']
		h1 = np.maximum(X @ self.params['w1'] + self.params['b1'], 0)
		h2 = np.maximum(h1 @ self.params['w2'] + self.params['b2'], 0)
		scores = h2 @ self.params['w3'] + self.params['b3']
		return scores

	def predict(self, X):
		return np.argmax(self.forward(X), axis=1)

def get_data(path):
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
	# 	-Get the data 				              #
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    data = pd.read_csv(path, sep='\s*,\s*',
                           header=0, encoding='ascii', engine='python')
    train_x = data.iloc[0:800,0:]
    train_x = np.asarray(train_x)
    train_y = train_x[0:800,0].astype('int')
    t_x = train_x[0:800,1:]
    validation = np.asarray(data.iloc[800:,0:])
    v_x = validation[:,1:]
    val_y = validation[:,0].astype('int')
    return t_x,train_y,v_x,val_y

def Normal(x):
    Fare_Norm = normalize(x[:,[0,2,5]],axis=0)
    Norm = np.concatenate((x[:,[1,3,4]] , Fare_Norm) ,axis = 1 )
    return Norm

def Standard(x):
	Scalar = StandardScaler()
	Scalar.fit(x)
	x_std = Scalar.transform(x)
	return x_std

def data_to_onehot(x):
	pc = x[:,0].astype('int')
	pc = pc-1
	n_values = np.max(pc) + 1
	pconehot = np.eye(n_values)[pc]
	onehotdata = np.concatenate((x[:,[1,2,3,4,5]] , pconehot),axis = 1)
	return onehotdata

def plotvorrelation(t_x_std,t_y,v_x_std,v_y,bars):
	corr_ = []
	y_pos = np.arange(len(bars))
	for i in range(6):		
		a = np.concatenate((t_x_std[:,i],v_x_std[:,i]),axis = None)
		v = np.concatenate((t_y,v_y))
		corr = np.asscalar(np.correlate(a,v))
		corr_.append(corr)
	plt.bar(y_pos , corr_ , alpha = 0.5 , align = 'center', label = 'Cross-correlation')
	plt.xticks(y_pos, bars)
	plt.legend(loc = 'best')
	plt.title("Correlation between feature and label")
	plt.tight_layout()
	plt.savefig("corr.png")
def Plotcovariancmatrix(t_x_std,bars):
	cov_mat = np.cov(t_x_std.T)
	y_pos = np.arange(len(bars))
	eigen_vals , eigen_vecs = np.linalg.eig(cov_mat)
	tot = sum(eigen_vals)
	var_exp = [(i/tot)for i in (eigen_vals)]
	cum_var_exp = np.cumsum(var_exp)
	
	plt.bar(y_pos , var_exp , alpha = 0.5 , align = 'center', label = ' explained variance')
	plt.step(range(0,6), cum_var_exp, where = 'mid',label = 'cumlative')
	plt.xticks(y_pos, bars)
	plt.legend(loc = 'best')
	plt.title("Covariance Matrix")
	plt.tight_layout()
	plt.savefig("covarianc.png")

def main():
	steps = 100000
	lr = 0.099
	decay = 0.97
	batch = 8
	mymodel = model(8,3,3,2,lr,decay,steps,batch)
	csv_path = 'titanic.csv'
	t_x,t_y,v_x,v_y = get_data(csv_path)
	t_x = data_to_onehot(t_x)
	v_x = data_to_onehot(v_x)
	Scalar = StandardScaler()
	Scalar.fit(t_x)
	pickle.dump(Scalar,open('scaler.p','wb'))
	t_x_std = Scalar.transform(t_x)
	v_x_std = Scalar.transform(v_x)
	#bars = ('Pclass', 'sex', 'Age', 'Sibsp', 'Parch','Fare')
	
	#plotvorrelation(t_x_std,t_y,v_x_std,v_y,bars)
	#Plotcovariancmatrix(t_x_std,bars)
	#t_x = Normal(t_x)
	
	#v_x = Normal(v_x)
	result = mymodel.train(t_x_std, t_y, v_x_std, v_y)
	
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.spines['top'].set_color('white')
	ax.spines['bottom'].set_color('white')
	ax.spines['left'].set_color('white')
	ax.spines['right'].set_color('white')
	ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
	ax1 = fig.add_subplot(211)
	ax2 = fig.add_subplot(212)
	ax1.plot(result['loss_history'][1:],'gx',label = "train")
	ax1.plot(result['val_loss_history'][1:],'r',label = "test" )
	ax1.set_title("Loss")
	ax1.legend(loc = "upper right")
	ax1.set_ylabel("Average cross entropy")
	ax2.plot(result['train_history'],'g',label = "train")
	ax2.plot(result['val_history'],'r',label = "test")
	ax2.legend(loc = "upper right")
	ax2.set_ylim((0,1))
	ax2.set_title("Error Rate")
	ax2.set_ylabel("Error rate")
	ax.set_xlabel("Number of epoch")
	plt.legend()
	plt.tight_layout()
	plt.savefig("Nodrmal.png")


if __name__ == "__main__":
	main()
