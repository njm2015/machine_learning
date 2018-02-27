import numpy as np
import math

epsilon = 1e-4

def lin_reg_h(X, theta):
	'''
		Simple multivariable hypothesis function for linear regression
		@param X - m x n numpy matrix
		@param theta - n x 1 numpy matrix
		@return - m x 1 numpy matrix, result of hypothesis function
	'''
	return X * theta

def sigmoid_h(X, theta):
	'''
		Sigmoid/logistic function used compute output of training examples
		@param X - m x n numpy matrix, training examples
		@param theta - n x 1 numpy matrix, feature parameters
		@return - m x 2 numpy matrix, hypothesis result
	'''
	return 1 / (1 + np.exp(-1 * lin_reg_h(X, theta)))

def lin_reg_cost(X, y, theta, alpha):
	'''
		Computes linear regression cost and also gradient
		@param X - m x n numpy matrix (m=training examples, n-1=features)
		@param y - m x 1 numpy matrix training set outputs
		@param theta - n x 1 numpy matrix, feature parameters to compute cost of
		@param alpha - learning rate (gradient)
		@return J - linear cost w.r. to feature parameters
		@return grad - n x 1 numpy matrix, gradient of feature parameters
	'''
	m = X.shape[0]
	err = lin_reg_h(X, theta) - y
	J = err.T * err
	grad = (alpha * (1/m)) * (X.T * err)
	return J, grad

def log_reg_cost(X, y, theta, alpha):
	'''
		Computes logistic regression cost and gradient
		@param X - m x n numpy matrix (m=training examples, n-1=features)
		@param y - m x 1 numpy matrix training set outputs
		@param theta - n x 1 numpy matrix, feature parameters to compute cost of
		@param alpha - learning rate (gradient)
		@return J - logistic cost w.r. to feature parameters
		@return grad - n x 1 numpy matrix, gradient of feature parameters
	'''
	m = X.shape[0]
	g = sigmoid_h(X,theta)
	J = (1/m)*(-1*y.T*np.log(g) - (1-y).T*np.log(1-g))
	grad = (alpha/m)*X.T*(g-y)
	return J, grad

def prep_data(X):
	return np.insert(X, 0, np.ones(X.shape[0]), axis=1)

def prep_theta(theta):
	return np.insert(theta, 0, np.matrix([0]), axis=0)

def fminunc(X, y, cost_func, learn_rate, initial_theta, num_iters):
	'''
		Mimics Matlab fminunc function to run gradient descent
		@param X - m x n numpy matrix (m=training examples, n-1=features)
		@param y - m x 1 numpy matrix training set outputs
		@param cost_func - cost function to use (see included functions)
		@param learn_rate - rate at which gradient descent operates
		@param initial_theta - n x 1 numpy matrix, starting state for feature parameters
		@param num_iters - number of iterations to run gradient descent algo
		@return - n x 1 numpy matrix, finalized feature parameters
	'''
	X = prep_data(X)
	ret_theta = prep_theta(initial_theta)
	J = int()
	for i in range(num_iters):
		J, grad = cost_func(X, y, ret_theta, learn_rate)
		ret_theta -= grad
	
	return ret_theta, J
