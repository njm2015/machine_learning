import gd
import numpy as np

if __name__ == '__main__':

	'''
	for i in range(1):
		X = np.matrix([[1,8,3,2,0],[1,4,1,1,3],[1,2,0,2,3]], dtype=np.float32)
		y = np.matrix([[65],[41],[26]], dtype=np.float32)
		theta = np.matrix(np.random.rand(5,1), dtype=np.float32)
		true_theta = np.matrix([[6],[7],[1],[0],[2]], dtype=np.float32)

		ret_theta, ret_J = gd.fminunc(X, y, gd.lin_reg_cost, 0.01, theta, 100000)
		print((ret_theta, ret_J))
		print('---')
		print(gd.lin_reg_h(X, ret_theta))

		
		#true_theta, true_J = gd.fminunc(X, y, gd.lin_reg_cost, 0.01, true_theta, 100000)

		#print(gd.lin_reg_h(X, true_theta))
		#print(gd.lin_reg_cost(X, y, true_theta, 0.01))

	'''

	X = np.matrix([[1,0],[1,1],[1,0],[1,1],[1,1],[1,0],[1,0]], dtype=np.float32)
	y = np.matrix([[0],[1],[0],[1],[1],[0],[0]], dtype=np.float32)
	theta = np.matrix([[0],[0]], dtype=np.float32)

	ret_theta, J = gd.fminunc(X, y, gd.log_reg_cost, 0.01, theta, 10000)

	print(ret_theta, J)

	print(gd.sigmoid_h(X, ret_theta))