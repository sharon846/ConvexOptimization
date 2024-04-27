from scipy.optimize import minimize
from scipy.optimize import Bounds
import numpy as np
import cvxpy as cp
from scipy.spatial.transform import Rotation as R

np.random.seed(42)
np.seterr(invalid='ignore')

def generate_positive_semidefinite_matrix(n):
	while True:
		A = np.random.rand(n, n)
		B = A.T @ A

		if is_positive_semidefinite(B) and np.all(B.T == B):
			return A,B

def is_positive_semidefinite(matrix):
    # Compute the eigenvalues of the matrix
    eigenvalues, _ = np.linalg.eig(matrix)
    
    # Check if all eigenvalues are non-negative
    if np.all(eigenvalues >= 0):
        return True
    else:
        return False


def is_rotation_matrix(X):
	return np.linalg.det(X) == 1 and np.all(np.linalg.inv(X) == X.T)

def f_q(q):
	return np.array([
			q[0]**2+q[1]**2-q[2]**2-np.sqrt(1-q[0]**2-q[1]**2-q[2]**2)**2, 2*q[1]*q[2]-2*q[0]*np.sqrt(1-q[0]**2-q[1]**2-q[2]**2), 2*q[1]*np.sqrt(1-q[0]**2-q[1]**2-q[2]**2)+2*q[0]*q[2],\
			2*q[1]*q[2]+2*q[0]*np.sqrt(1-q[0]**2-q[1]**2-q[2]**2), q[0]**2+q[2]**2-q[1]**2-np.sqrt(1-q[0]**2-q[1]**2-q[2]**2)**2, 2*q[2]*np.sqrt(1-q[0]**2-q[1]**2-q[2]**2)-2*q[0]*q[1],\
			2*q[1]*np.sqrt(1-q[0]**2-q[1]**2-q[2]**2)-2*q[0]*q[2], 2*q[2]*np.sqrt(1-q[0]**2-q[1]**2-q[2]**2)+2*q[0]*q[1], q[0]**2+np.sqrt(1-q[0]**2-q[1]**2-q[2]**2)**2-q[1]**2-q[2]**2])

def objective_function(q):
	return -np.sum(A.dot(f_q(q))**2)

# Initial guess for the optimization algorithm
initial_guess = [0.5, 0.5, 0.5]

# Generate a random 9x9 matrix A and a random vector b
A ,B = generate_positive_semidefinite_matrix(9)

bounds = Bounds([-1, -1, -1], [1, 1, 1])
# Perform optimization
result = minimize(objective_function, initial_guess, method='SLSQP', bounds=bounds)

#print(A)

# Extract the optimal solution
optimal_solution = result.x
optimal_value = result.fun
q = [optimal_solution[0], optimal_solution[1], optimal_solution[2]]
q.append(np.sqrt(1 - np.linalg.norm(optimal_solution) ** 2))

C = np.outer(f_q(q), f_q(q))

print("Optimal value:", -optimal_value)
print("Optimal solution:", q)
print("Optimal solution norm:", np.linalg.norm(q))

print("------------------------------")
print("Doing refinment")
print("------------------------------")

X = cp.Variable((9,9), PSD=True)				# Y = y^t*y and x = f(q)		# PSD=True

objective = cp.Minimize(0)
constraints = [X >> 0, cp.trace(X) == 3, cp.norm(X - C, "fro") <= 1e-11, X << C]			

problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.CVXOPT)

print(f'solution is optimal: {problem.status == cp.OPTIMAL}')
print(f'positive semideinite (X): {is_positive_semidefinite(X.value)}')
print(f'rank (X) is {np.linalg.matrix_rank(X.value)}')

U, Sigma, VT = np.linalg.svd(X.value)

x = U[:, 0] * np.sqrt(Sigma[0]) 
x *= -1
# print(f'distance between x (decomposed) and X is: {np.linalg.norm(np.outer(x, x) - X.value)}')
# print("------------------------------")
# print("Rotation matrix found:")
# print(x.reshape(3,3))
# print("------------------------------")

# print("------------------------------")
# print("Rotation matrix from first solution")
# r = R.from_quat(q).as_matrix()
# print(r)
# print("------------------------------")

w = 0.5 * np.sqrt(1 + x[0] + x[4] + x[8])
q2 = [w, (x[7] - x[5]) / (4*w), (x[2] - x[6]) / (4*w), (x[3] - x[1]) / (4*w)]

print("Optimal value:", -objective_function(q2))
print("Optimal solution:", q2)
print("Optimal solution norm:", np.linalg.norm(q2))

