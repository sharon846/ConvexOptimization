import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

A = np.array([[3,-1,1], [2,1,2], [1,2.5,-1]])
A0 = A.T @ A


b = np.array([2,-1,3])
b0 = -A.T @ b 

c0 = b.T @ b

mat = np.zeros((4,4))
mat[:3,:3] = A0
mat[3,:3] = b0.T
mat[:3, 3] = b0
mat[3,3] = c0

gamma = cp.Variable()
lamda = cp.Variable()


matrix = cp.bmat([[mat[0,0] + lamda, mat[0,1], mat[0,2], mat[0,3]],
				  [mat[1,0], mat[1,1] + lamda, mat[1,2], mat[1,3]],
				  [mat[2,0], mat[2,1], mat[2,2] + lamda, mat[2,3]],
				  [mat[3,0], mat[3,1], mat[3,2], mat[3,3] - lamda - gamma]])


objective = cp.Maximize(gamma)
constraints = [matrix >> 0, lamda >= 0]

problem = cp.Problem(objective, constraints)

problem.solve(solver=cp.SCS)

print(problem.value)