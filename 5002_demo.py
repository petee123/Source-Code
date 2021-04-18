
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers, regularizers
import warnings
warnings.filterwarnings("ignore")
tf.compat.v1.disable_eager_execution()
import statistics
from sklearn.model_selection import train_test_split

# define global variable  
# path, num_op_unary, num_op_binary, max_order, num_batch, optimizer, lr, epochs, evaluate, task, dataset, model, alpha, lr_value, RL_model, reg, controller, num_random_sample, lambd, multiprocessing, package, method, origin_result, num_process, infos
# Moment 
# global variable assign the default value outside the main
dataset = "BMI"
#path = './dataset/' + dataset + '.csv'
path = dataset + '.csv'
num_op_unary = 3
num_op_binary = 5
max_order = 5
num_batch = 32
optimizer = 'adam'
lr = 0.01
epochs = 10
evaluate = 'r2'
task = 'regression'
model = 'RF'
alpha = 0.99
lr_value = 1e-3
RL_model = 'PG'
reg = 1e-5
controller = 'rnn'
num_random_sample = 5
lambd = 0.4
package = 'sklearn'
method = 'train'
origin_result, method, name = 1, "train", None
num_process, infos = 24, []
multiprocessing=False
 
class Controller:
	
	def __init__(self, num_op_unary, num_op_binary, max_order, num_batch, optimizer, 
		 lr, epochs, evaluate, task, dataset, 
		 model, alpha, lr_value, RL_model, reg, 
		 controller, num_random_sample, lambd, multiprocessing, package, num_feature):
		self.num_feature = num_feature
		self.num_op_unary = num_op_unary
		self.num_op_binary = num_op_binary
		self.num_op = num_op_unary + (self.num_feature-1)*num_op_binary + 1
		self.max_order = max_order
		self.num_batch = num_batch
		self.opt = optimizer
		self.lr = lr
		self.lr_value = lr_value
		self.num_action = self.num_feature * self.max_order
		self.reg = reg

	
	def _create_rnn(self):
		self.rnns = {}
		for i in range(self.num_feature):
			self.rnns['rnn%d'%i] = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(num_units=self.num_op, name='rnn%d'%i)
			# self.rnns['rnn%d'%i] = tf.compat.v1.contrib.rnn.BasicLSTMCell(num_units=self.num_op, name='rnn%d'%i)

	def _create_placeholder(self):
		self.concat_action = tf.compat.v1.placeholder(tf.compat.v1.int32, shape=[self.num_batch,self.num_action], name='concat_action')
		# self.concat_action = tf.compat.v1.placeholder(tf.compat.v1.int32, shape=[self.num_batch,self.num_action], name='concat_action')

		self.rewards = tf.compat.v1.placeholder(tf.compat.v1.float32, shape=[self.num_batch,self.num_action], name='rewards')

		self.state = tf.compat.v1.placeholder(tf.compat.v1.int32, shape=[None,self.num_action], name='state')
		self.value = tf.compat.v1.placeholder(tf.compat.v1.float32, shape=[None,1], name='value')


	def _create_variable(self):
		self.input0 = np.ones(shape=[self.num_feature,self.num_op], dtype=np.float32)
		self.input0 = self.input0 / self.num_op

		self.value_estimator = Sequential([
			Dense(64, input_dim=self.num_action, kernel_regularizer=regularizers.l2(self.reg)),
			Activation('tanh'),
			Dense(16, kernel_regularizer=regularizers.l2(self.reg)),
			Activation('tanh'),
			Dense(4, kernel_regularizer=regularizers.l2(self.reg)),
			Activation('tanh'),
			Dense(1)])
		self.value_optimizer = optimizers.Adam(lr=self.lr_value)
		self.value_estimator.compile(
			optimizer=self.value_optimizer, loss='mean_squared_error')

	def _create_inference(self):
		self.outputs = {}
		
		for i in range(self.num_feature):
			tmp_h = self.rnns['rnn%d'%i].zero_state(1, tf.compat.v1.float32)
			tmp_input = tf.compat.v1.reshape(tf.compat.v1.nn.embedding_lookup(self.input0, i),
				[1,-1])
			for order in range(self.max_order):
				tmp_input, tmp_h = self.rnns['rnn%d'%i].__call__(tmp_input, tmp_h)
				if order == 0:
					self.outputs['output%d'%i] = tmp_input
				else:
					self.outputs['output%d'%i] = tf.concat([self.outputs['output%d'%i], tmp_input], axis=0)
		self.concat_output = tf.concat(list(self.outputs.values()), axis=0, name='concat_output')	


	def _create_loss(self):
		self.loss = 0.0
		for batch_count in range(self.num_batch):
			action = tf.compat.v1.nn.embedding_lookup(self.concat_action, batch_count)	
			reward = tf.compat.v1.nn.embedding_lookup(self.rewards, batch_count)	
			action_index = tf.compat.v1.stack([list(range(self.num_action)), action], axis=1)
			action_probs = tf.compat.v1.squeeze(tf.compat.v1.nn.softmax(self.concat_output))
			pick_action_prob = tf.compat.v1.gather_nd(action_probs, action_index)
			loss_batch = tf.compat.v1.reduce_sum((-tf.compat.v1.log(pick_action_prob)) * reward)
			loss_entropy = tf.compat.v1.reduce_sum(-action_probs * tf.compat.v1.log(action_probs)) * self.reg
			loss_reg = 0.0
			for i in range(self.num_feature):
				weights = self.rnns['rnn%d'%i].weights
				for w in weights:
					loss_reg += self.reg * tf.compat.v1.reduce_sum(tf.compat.v1.square(w))	
			self.loss += loss_batch + loss_entropy + loss_reg
		
		self.loss /= self.num_batch


	def _create_optimizer(self):
		if self.opt == 'adam':
			self.optimizer = tf.compat.v1.train.AdamOptimizer(
				learning_rate=self.lr).minimize(self.loss)
			
		elif self.opt == 'adagrad':
			self.optimizer = tf.compat.v1.train.AdagradOptimizer(
				learning_rate=self.lr).minimize(self.loss)
			

	def build_graph(self):
		self._create_rnn()
		self._create_variable()
		self._create_placeholder()
		self._create_inference()
		self._create_loss()
		self._create_optimizer()

	def update_policy(self, feed_dict, sess=None):
		_, loss = sess.run([self.optimizer,self.loss], feed_dict=feed_dict)
		return loss

	def update_value(self, state, value, sess=None):
		self.value_estimator.fit(state, value, epochs=20, batch_size=32, verbose=0)


	def predict_value(self, state, sess=None):
		value = self.value_estimator.predict(state)
		return np.squeeze(value)


"""# utils.py"""

from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn import svm
import numpy as np
import pandas as pd
import sys
import os
import logging

def mod_column(c1, c2):
	r = []
	for i in range(c2.shape[0]):
		if c2[i] == 0:
			r.append(0)
		else:
			r.append(np.mod(c1[i],c2[i]))
	return r


def evaluate1(X, y, num_op_unary, num_op_binary, max_order, num_batch, optimizer, lr, epochs, evaluate, task, dataset, model, alpha, lr_value, RL_model, reg, controller, num_random_sample, lambd, multiprocessing, package):

	if task == 'regression':
		if model == 'LR':
			model = Lasso()
		elif model == 'RF':
			model = RandomForestRegressor(n_estimators=10, random_state=0)
			
		if evaluate == 'mae':
			r_mae = cross_val_score(model, X, y, cv=5, 
				scoring='neg_mean_absolute_error').mean()
			return r_mae
		elif evaluate == 'mse':
			r_mse = cross_val_score(model, X, y, cv=5, 
				scoring='neg_mean_squared_error').mean()
			return r_mse
		elif evaluate == 'r2':
			r_r2 = cross_val_score(model, X, y, cv=5).mean()
			return r_r2
		elif evaluate == 'rae':
			print("rae")
			y_mean = statistics.mean(y)
			X1 = X.copy()
			y1 = y.copy()
			r_rae = 0
			Num = len(X.index)
			Seg = int(Num / 5)
			
			for i in range(5):
				if (i == 0):
					X_test = X1[i*Seg:Seg]
					X_train = X1[Seg:]

					y_test = y1[i*Seg:Seg]
					y_train = y1[Seg:]

				elif (i == 4):
					X_test = X1[i*Seg:]
					X_train = X1[:i*Seg]

					y_test = y1[i*Seg:]
					y_train = y1[:i*Seg]

				else:
					X_test = X1[i*Seg:(i+1)*Seg]
					y_test = y1[i*Seg:(i+1)*Seg]

					X_train = X1[:i*Seg]
					y_train = y1[:i*Seg]

					X_train2 = X1[(i+1)*Seg:]
					y_train2 = y1[(i+1)*Seg:]

					X_train = X_train.append(X_train2)
					y_train = y_train.append(y_train2)


			
				model.fit(X_train, y_train)

				y_pred = model.predict(X_test)
				diff1 = y_pred - y_test
				diff2 = y_mean - y_test
 
				diff1_sum = 0
				for i in diff1:
					diff1_sum = diff1_sum + abs(i)
 
				diff2_sum = 0
				for i in diff2:
					diff2_sum = diff2_sum + abs(i)
 
				r_rae = 1 - (diff1_sum/diff2_sum) + r_rae

			return (r_rae/5)
 

		

	elif task == 'classification':
		le = LabelEncoder()
		y = le.fit_transform(y)

		if model == 'RF':
			model = RandomForestClassifier(n_estimators=10, random_state=0)
		elif model == 'LR':
			model = LogisticRegression(multi_class='ovr')
		elif model == 'SVM':
			model = svm.SVC()
		if evaluate == 'f_score':
			s = cross_val_score(model, X, y, scoring='f1', cv=5).mean()
		elif evaluate == 'auc':
			model = RandomForestClassifier(max_depth=10, random_state=0)
			split_pos = X.shape[0] // 10
			X_train, X_test = X[:9*split_pos], X[9*split_pos:]
			y_train, y_test = y[:9*split_pos], y[9*split_pos:]
			model.fit(X_train, y_train)
			y_pred = model.predict_proba(X_test)
			s = evaluate_(y_test, y_pred)
		return s

def evaluate_(y_true, y_pred):
	num_class = max(y_true) + 1
	y_true = np.eye(num_class)[y_true]
	return 2 * roc_auc_score(y_true, y_pred) - 1

def init_name_and_log(num_op_unary, num_op_binary, max_order, num_batch, optimizer, lr, epochs, evaluate, task, dataset, model, alpha, lr_value, RL_model, reg, controller, num_random_sample, lambd, multiprocessing, package):
	name = dataset + '_' + controller + '_' + RL_model + '_' + \
		model + '_' + package + '_' + evaluate + '_' + \
		str(num_batch) + '_' + str(num_random_sample) + '_' + \
		str(max_order) + '_' + optimizer + str(lr) + '_' + \
		str(lr_value) + '_' + str(alpha) + '_' + str(lambd)

	if not os.path.exists('log'):
		os.mkdir('log')
	for handler in logging.root.handlers[:]:
		logging.root.removeHandler(handler)
	logging.basicConfig(filename='log/' + name + '.log',
		level=logging.INFO)
	logging.info('--start--')
	return name

def save_result(infos, name):
	if not os.path.exists('result'):
		os.mkdir('result')
	save_path = 'result/' + name + '.txt'

	with open(save_path, 'w') as f:
		for info in infos:
			f.write(str(info) + '\n')
	print(name, 'saved')

import logging
# from Controller import Controller, Controller_sequence, Controller_pure
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")
import argparse
from multiprocessing import Pool, cpu_count, Process
import multiprocessing
# from utils import mod_column, evaluate, init_name_and_log, save_result
from collections import ChainMap
from subprocess import Popen, PIPE
from time import time, sleep
import os
# from Java_service import start_service_pool, stop_service_pool, find_free_port
# import rpyc
import random
import tensorflow as tf



def get_reward(actions):
	global path, num_op_unary, num_op_binary, max_order, num_batch, optimizer, lr, epochs, evaluate, task, dataset, model, alpha, lr_value, RL_model, reg, controller, num_random_sample, lambd, multiprocessing, package, method, origin_result
	
	print("################## get_reward start ##################")
	print ("path: " + path)
	X = pd.read_csv(path)
	num_feature = X.shape[1] - 1
	action_per_feature = int(len(actions) / num_feature)
	copies, copies_run, rewards = {}, [], []

	try:
		for feature_count in range(num_feature):
			feature_name = X.columns[feature_count]
			print("feature_name is ", feature_name)
			feature_actions = actions[feature_count*action_per_feature: (feature_count+1)*action_per_feature]
			copies[feature_count] = []
			if feature_actions[0] == 0:
				continue
			else:
				copy = np.array(X[feature_name].values)	

			print("num_op_unary is ", num_op_unary) 
			print("feature_count is ", feature_count)
			print("feature value is ", ' '.join(map(str, copy)))
   
			for action in feature_actions:
				print("feature action is ", action)
				if action == 0:
					break
										  
				#Billy test (work without error) 
				elif action > 0 and action <= num_op_unary:
					action_unary = action - 1
					if action_unary == 0:
						copy = np.squeeze(np.sqrt(abs(copy)))
					elif action_unary == 1:
						copy = np.squeeze(np.square(abs(copy)))
					elif action_unary == 2:
						scaler = MinMaxScaler()
						copy = np.squeeze(scaler.fit_transform(np.reshape(copy,[-1,1])))
					elif action_unary == 3:
						while (np.any(copy == 0)):
							copy = copy + 1e-5
						copy = np.squeeze(np.log(abs(np.array(copy))))
					elif action_unary == 4:
						while (np.any(copy == 0)):
							copy = copy + 1e-5
						copy = np.squeeze(1 / (np.array(copy))) 
						
				else:
					action_binary = (action-num_op_unary-1) // (num_feature-1)
					rank = np.mod(action-num_op_unary-1, num_feature-1)

					if rank >= feature_count:
						rank += 1
					target_feature_name = X.columns[rank]
					target = np.array(X[target_feature_name].values) 

					if action_binary == 0:
						copy = np.squeeze(copy + target)
					elif action_binary == 1:
						copy = np.squeeze(copy - target)
					elif action_binary == 2:
						copy = np.squeeze(copy * target)
					elif action_binary == 3:
						while (np.any(target == 0)):
							target = target + 1e-5
						copy = np.squeeze(copy / target) 
					elif action_binary == 4:
						copy = np.squeeze(mod_column(copy, X[target_feature_name].values))

				copies[feature_count].append(copy)
			copies_run.append(copy)

		print("Finish feature action looping")
		print("method is: ", method)
		if method == 'train':
			print("this round train original result: ", origin_result)
			former_result = origin_result
			former_copys = [None]
			for key in sorted(copies.keys()):
				reward, former_result, return_copy = get_reward_per_feature( 
					copies[key], action_per_feature, former_result, former_copys)
				former_copys.append(return_copy)
				rewards += reward
				print("reward in after training process: ", reward)
			
			print(" rewards result is : ", ' '.join(map(str, rewards)))
			print("############################## get reward end ##############################")
			return rewards

		elif method == 'test':
			for i in range(len(copies_run)):
				X.insert(0, 'new%d'%i, copies_run[i])
				
			if package == 'sklearn':
				y = X[X.columns[-1]]
				del X[X.columns[-1]]
				result = evaluate1(X, y, num_op_unary, num_op_binary, max_order, num_batch, optimizer, lr, epochs, evaluate, task, dataset, model, alpha, lr_value, RL_model, reg, controller, num_random_sample, lambd, multiprocessing, package)
				print("result after test process: ", result)
			print("############################## get reward end ##############################")		   
			return result
	except RuntimeError as e:
		print("RuntimeError: ", e)

def get_reward_per_feature(copies, count, former_result, former_copys=[None]):
	global path, num_op_unary, num_op_binary, max_order, num_batch, optimizer, lr, epochs, evaluate, task, dataset, model, alpha, lr_value, RL_model, reg, controller, num_random_sample, lambd, multiprocessing, package, origin_result
	print ("path: " + path)
	X = pd.read_csv(path)
	if package == 'sklearn':
		y = X[X.columns[-1]]
		del X[X.columns[-1]]

	reward = []
	previous_result = former_result
	for i,former_copy in enumerate(former_copys):
		if not former_copy is None:
			X.insert(0, 'former%d'%i, former_copy)

	for copy in copies:
		X.insert(0, 'new', copy)
		if package == 'sklearn':
			current_result = evaluate1(X, y, num_op_unary, num_op_binary, max_order, num_batch, optimizer, lr, epochs, evaluate, task, dataset, model, alpha, lr_value, RL_model, reg, controller, num_random_sample, lambd, multiprocessing, package)
		if(previous_result is None):
			reward.append(current_result)
		else:
			reward.append(current_result - previous_result)
		previous_result = current_result
		del X['new']

	reward_till_now = len(reward)
	for _ in range(count - reward_till_now):
		reward.append(0)
	if len(copies) == 0:
		return_copy = None
	else:
		return_copy = copies[-1]

	return reward, previous_result, return_copy

def random_run(nnum_random_sample, nmodel, l=None, p=None):
	global num_op_unary, num_op_binary, max_order, num_batch, optimizer, lr, epochs, evaluate, task, dataset, model, alpha, lr_value, RL_model, reg, controller, num_random_sample, lambd, multiprocessing, package, num_process, origin_result
	samples = []
	for i in range(nnum_random_sample):
		sample = []
		for _ in range(nmodel.num_action):
			sample.append(np.random.randint(nmodel.num_op))
		samples.append(sample)

	if multiprocessing:	
		if package == 'sklearn':
			pool = Pool(num_process)	
		res = list(pool.map(get_reward, samples))
		pool.close()
		pool.join()
	else:
		res = []
		for sample in samples:
			res.append(get_reward(sample))

	random_result = max(res)
	random_sample = samples[res.index(random_result)]

	return random_result, random_sample
	
def train(nmodel, l=None, p=None):
	global path, num_op_unary, num_op_binary, max_order, num_batch, optimizer, lr, epochs, evaluate, task, dataset, model, alpha, lr_value, RL_model, reg, controller, num_random_sample, lambd, multiprocessing, package, infos, method, origin_result, num_process
	print ("path: " + path)
	X = pd.read_csv(path)
	print(X)
	if package == 'sklearn':
		y = X[X.columns[-1]]
		del X[X.columns[-1]]
		print(X.shape)

		origin_result = evaluate1(X, y, num_op_unary, num_op_binary, max_order, num_batch, optimizer, lr, epochs, evaluate, task, dataset, model, alpha, lr_value, RL_model, reg, controller, num_random_sample, lambd, multiprocessing, package)	
	best_result = origin_result
	print("origin_result is: ", origin_result)
 
	print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
 
	gpus = tf.config.list_physical_devices('GPU')
	if gpus:
	# Restrict TensorFlow to only use the first GPU
		try:
			tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
			logical_gpus = tf.config.experimental.list_logical_devices('GPU')
			tf.config.experimental.set_memory_growth(gpus[0], True)
			print("There are ", len(gpus), " Physical GPUs,", " and ", len(logical_gpus), " Logical GPU")
		except RuntimeError as e: # Visible devices must be set before GPUs have been initialized
			print(e)
   
	opts = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.4)
	#config = tf.compat.v1.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1, gpu_options=opts)
	config = tf.compat.v1.ConfigProto(gpu_options=opts)

	with tf.compat.v1.Session(config=config) as sess:
		init_op = tf.compat.v1.group(tf.compat.v1.global_variables_initializer(), 
			tf.compat.v1.local_variables_initializer())
		sess.run(init_op)
		
		nmodel_result = -10000.0 
		train_set, values = [], []
		for epoch_count in range(epochs):
			concat_action = []
			probs_action = sess.run(tf.compat.v1.nn.softmax(nmodel.concat_output))
			for batch_count in range(num_batch):
				batch_action = []
				for i in range(probs_action.shape[0]):
					sample_action = np.random.choice(len(probs_action[i]), p=probs_action[i])
					batch_action.append(sample_action)
				concat_action.append(batch_action)
					
			method = 'train'
			if multiprocessing:
				if package == 'sklearn':
					pool = Pool(num_process)		
				rewards = np.array(pool.map(get_reward, concat_action))
				pool.close()
				pool.join()
			else:
				rewards = []
				for action in concat_action:
					rewards.append(get_reward(action))
				rewards = np.array(rewards)

			method = 'test'
			if multiprocessing:
				if package == 'sklearn':
					pool = Pool(num_process)	  
				results = pool.map(get_reward, concat_action)
				pool.close()
				pool.join()
			else:
				results = []
				for action in concat_action:
					results.append(get_reward(action))
			nmodel_result = max(nmodel_result, max(results))


			if RL_model == 'AC':
				target_set = []
				for batch_count in range(num_batch):
					action = concat_action[batch_count]
					for i in range(nmodel.num_action):
						train_tmp = list(np.zeros(nmodel.num_action, dtype=int))
						target_tmp = list(np.zeros(nmodel.num_action, dtype=int))
						
						train_tmp[0:i] = list(action[0:i])
						target_tmp[0:i+1] = list(action[0:i+1])

						train_set.append(train_tmp)
						target_set.append(target_tmp)

				state = np.reshape(train_set, [-1,nmodel.num_action])
				next_state = np.reshape(target_set, [-1,nmodel.num_action])

				value = nmodel.predict_value(next_state) * alpha + rewards.flatten()
				values += list(value)
				nmodel.update_value(state, values)

				rewards_predict = nmodel.predict_value(next_state) * alpha - \
					nmodel.predict_value(state[-np.shape(next_state)[0]:]) + rewards.flatten()
				rewards = np.reshape(rewards_predict, [num_batch,-1])

			
			elif RL_model == 'PG':
				for i in range(nmodel.num_action):
					base = rewards[:,i:]
					rewards_order = np.zeros_like(rewards[:,i])
					for j in range(base.shape[1]):
						order = j + 1
						base_order = base[:,0:order]
						alphas = []
						for o in range(order):
							alphas.append(pow(alpha, o))
						base_order = np.sum(base_order*alphas, axis=1)
						base_order = base_order * np.power(lambd, j) 
						rewards_order = rewards_order.astype(float) 
						rewards_order += base_order.astype(float) 
					rewards[:,i] = (1-lambd) * rewards_order
				

			feed_dict = {nmodel.concat_action: np.reshape(concat_action, [num_batch,-1]), \
				nmodel.rewards: np.reshape(rewards,[num_batch,-1])}
			loss_epoch = nmodel.update_policy(feed_dict, sess)


			method = 'test'
			probs_action = sess.run(tf.compat.v1.nn.softmax(nmodel.concat_output))
			best_action = probs_action.argmax(axis=1)
			nmodel_result = max(nmodel_result, get_reward(best_action))
			random_result, random_sample = random_run(num_random_sample, nmodel, l, p)

			best_result = max(best_result, nmodel_result)

#			global path, num_op_unary, num_op_binary, max_order, num_batch, optimizer, lr, epochs, evaluate, task, dataset, model, alpha, lr_value, RL_model, reg, controller, num_random_sample, lambd, multiprocessing, package, infos, method, origin_result, num_process
			
			print('Epoch-' + str(epoch_count+1))
			print('loss-' + str(loss_epoch))
			print('origin_result-' + str(origin_result))
			print('lr-' + str(lr))
			print('model_result-' + str(nmodel_result))
			print('best_action-' + str(best_action))
			print('best_result-' + str(best_result))
			print('random_result-' + str(random_result))
			print('random_sample-' + str(random_sample))
			
			
			
			# % (epoch_count+1, loss_epoch, origin_result, lr, nmodel_result, str(best_action), best_result, random_result, str(random_sample)))
			logging.info('Epoch %d: loss = %.4f, origin_result = %.4f, lr = %.3f, \n model_result = %.4f, best_action = %s, \n best_result = %.4f, random_result = %.4f, random_sample = %s' )
			# % (epoch_count+1, loss_epoch, origin_result, lr, nmodel_result, str(best_action), best_result, random_result, str(random_sample)))

			info = [epoch_count, loss_epoch, origin_result, nmodel_result, random_result]
			infos.append(info)

def main(pnum_op_unary=4, pnum_op_binary=5, pmax_order=5, pnum_batch=32, poptimizer='adam', 
		 plr=0.01, pepochs=10, pevaluate='r2', ptask='regression', pdataset='airfoil', 
		 pmodel='RF', palpha=0.99, plr_value=1e-3, pRL_model='PG', preg=1e-5, 
		 pcontroller='rnn', pnum_random_sample=5, plambd=0.4, pmultiprocessing=True, ppackage='sklearn'):
	global path, num_op_unary, num_op_binary, max_order, num_batch, optimizer, lr, epochs, evaluate, task, dataset, model, alpha, lr_value, RL_model, reg, controller, num_random_sample, lambd, multiprocessing, package, method, origin_result, num_process, infos	
	dataset = pdataset
	path = './dataset/' + dataset + '.csv'
	#path = dataset + '.csv'
	num_op_unary = pnum_op_unary
	num_op_binary = pnum_op_binary
	max_order = pmax_order
	num_batch = pnum_batch
	optimizer = poptimizer
	lr = plr
	epochs = pepochs
	evaluate = pevaluate
	task = ptask
	dataset = pdataset
	model = pmodel
	alpha = palpha
	lr_value = plr_value
	RL_model = pRL_model
	reg = preg
	controller = pcontroller
	num_random_sample = pnum_random_sample
	lambd = plambd
	multiprocessing = pmultiprocessing
	package = ppackage

	origin_result, method, name = None, None, None
	num_process, infos = 24, []
	name = init_name_and_log(num_op_unary, num_op_binary, max_order, num_batch, optimizer, lr, epochs, evaluate, task, dataset, model, alpha, lr_value, RL_model, reg, controller, num_random_sample, lambd, multiprocessing, package)
	print("name: ", name)

	num_feature = pd.read_csv(path).shape[1] - 1
	if controller == 'rnn':
		controller = Controller(num_op_unary, num_op_binary, max_order, num_batch, optimizer, 
		 lr, epochs, evaluate, task, dataset, 
		 model, alpha, lr_value, RL_model, reg, 
		 controller, num_random_sample, lambd, multiprocessing, package, num_feature)
  
	controller.build_graph()

	train(controller)

	save_result(infos, name)
	
	print("end")


# need to implement 1-rae

from datetime import datetime

# BMI
if __name__ == '__main__':
	
	
	
	s_now = datetime.now()
	S_timestamp = datetime.timestamp(s_now)
	print("Start Time =", s_now)

	#!/usr/bin/env python
	import psutil
	# gives a single float value
	print(psutil.cpu_percent())
	# gives an object with many fields
	print(psutil.virtual_memory())
	# you can convert that object to a dictionary 
	print(dict(psutil.virtual_memory()._asdict()))
	# you can have the percentage of used RAM
	print(psutil.virtual_memory().percent)
	# you can calculate percentage of available memory
	print(psutil.virtual_memory().available * 100 / psutil.virtual_memory().total)
	
	
	multiprocessing.freeze_support()
	#main(3, 4, 5, 32, 'adam', 0.01, 1, 'r2', 'regression', 'BMI', 'RF', 0.99, 1e-3, 'PG', 1e-5, 'rnn', 5, 0.4, False, 'sklearn')
	#main(5, 5, 5, 32, 'adam', 0.01, 1, 'rae', 'regression', 'BMI', 'RF', 0.99, 1e-3, 'PG', 1e-5, 'rnn', 5, 0.4, False, 'sklearn')

	
	main(5, 5, 5, 32, 'adam', 0.01, 16, 'rae', 'regression', 'BMI', 'RF', 0.99, 1e-3, 'PG', 1e-5, 'rnn', 5, 0.4, False, 'sklearn')
	
	
	now = datetime.now()
	E_timestamp = datetime.timestamp(now)
	print("Start Time =", s_now)
	print("End Time =", now)
	Total = E_timestamp - S_timestamp
	print("Total Time = " + str(Total))


	# gives a single float value
	print(psutil.cpu_percent())
	# gives an object with many fields
	print(psutil.virtual_memory())
	# you can convert that object to a dictionary 
	print(dict(psutil.virtual_memory()._asdict()))
	# you can have the percentage of used RAM
	print(psutil.virtual_memory().percent)
	# you can calculate percentage of available memory
	print(psutil.virtual_memory().available * 100 / psutil.virtual_memory().total)
	
# airfoil

# main(5, 4, 5, 32, 'adam', 0.01, 2, 'r2', 'regression', 'airfoil', 'RF', 0.99, 1e-3, 'PG', 1e-5, 'rnn', 5, 0.4, True, 'sklearn')