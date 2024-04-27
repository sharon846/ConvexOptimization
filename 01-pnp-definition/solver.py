import cvxpy as cp
import numpy as np


# Set a random seed for reproducibility
np.random.seed(42)

def generate_invertible_matrix(n):
	while True:
		# Generate a random n x n matrix
		matrix = np.random.rand(n, n)

		# Check if the matrix is invertible
		if np.linalg.matrix_rank(matrix) == n:
			# Matrix is invertible
			return matrix

def f_q(q):
	return cp.vstack([
		(q[0]**2+q[1]**2-q[2]**2-q[3]**2) / (q[0]**2+q[1]**2+q[2]**2+q[3]**2), 
		(2*q[1]*q[2]-2*q[0]*q[3]) / (q[0]**2+q[1]**2+q[2]**2+q[3]**2), 
		(2*q[1]*q[3]+2*q[0]*q[2]) / (q[0]**2+q[1]**2+q[2]**2+q[3]**2),
		(2*q[1]*q[2]+2*q[0]*q[3]) / (q[0]**2+q[1]**2+q[2]**2+q[3]**2), 
		(q[0]**2+q[2]**2-q[1]**2-q[3]**2) / (q[0]**2+q[1]**2+q[2]**2+q[3]**2), 
		(2*q[2]*q[3]-2*q[0]*q[1]) / (q[0]**2+q[1]**2+q[2]**2+q[3]**2),
		(2*q[1]*q[3]-2*q[0]*q[2]) / (q[0]**2+q[1]**2+q[2]**2+q[3]**2), 
		(2*q[2]*q[3]+2*q[0]*q[1]) / (q[0]**2+q[1]**2+q[2]**2+q[3]**2), 
		(q[0]**2+q[3]**2-q[1]**2-q[2]**2) / (q[0]**2+q[1]**2+q[2]**2+q[3]**2)])

# Generate a random 9x9 matrix A and a random vector b
A = generate_invertible_matrix(9)
#A = cp.Variable((9, 9), symmetric=True)

# Define optimization variable
q = cp.Variable(4)

objective = 21.52*q[0]**4 + 18.96*q[0]**3*q[1] + 1.52*q[0]**3*q[2] - 17.76*q[0]**2*q[1]**2 + 27.04*q[0]**2*q[1]*q[2] + 24*q[0]**2*q[1]*cp.sqrt(-q[0]**2 - q[1]**2 - q[2]**2 + 1) - 12.76*q[0]**2*q[2]**2 + 24*q[0]**2*q[2]*cp.sqrt(-q[0]**2 - q[1]**2 - q[2]**2 + 1) - 8*q[0]**2 - 30.64*q[0]*q[1]**3 - 8.8*q[0]*q[1]**2*q[2] + 16*q[0]*q[1]**2*cp.sqrt(-q[0]**2 - q[1]**2 - q[2]**2 + 1) - 1.28*q[0]*q[1]*q[2]**2 + 32*q[0]*q[1]*q[2]*cp.sqrt(-q[0]**2 - q[1]**2 - q[2]**2 + 1) + 16*q[0]*q[1]*(-q[0]**2 - q[1]**2 - q[2]**2 + 1) - 12*q[0]*q[1] - 8*q[0]*q[2]**3 + 16*q[0]*q[2]**2*cp.sqrt(-q[0]**2 - q[1]**2 - q[2]**2 + 1) + 16*q[0]*q[2]*(-q[0]**2 - q[1]**2 - q[2]**2 + 1) - 12*q[0]*q[2] - 4*q[0]*cp.sqrt(-q[0]**2 - q[1]**2 - q[2]**2 + 1) - 10.68*q[1]**4 - 29.52*q[1]**3*q[2] + 16*q[1]**3*cp.sqrt(-q[0]**2 - q[1]**2 - q[2]**2 + 1) + 38.32*q[1]**2*q[2]**2 + 48*q[1]**2*q[2]*cp.sqrt(-q[0]**2 - q[1]**2 - q[2]**2 + 1) + 16*q[1]**2*(-q[0]**2 - q[1]**2 - q[2]**2 + 1) - 4*q[1]**2 + 48*q[1]*q[2]**2*cp.sqrt(-q[0]**2 - q[1]**2 - q[2]**2 + 1) + 32*q[1]*q[2]*(-q[0]**2 - q[1]**2 - q[2]**2 + 1) - 8*q[1]*q[2] + 8*q[1]*cp.sqrt(-q[0]**2 - q[1]**2 - q[2]**2 + 1) + 16*q[2]**3*cp.sqrt(-q[0]**2 - q[1]**2 - q[2]**2 + 1) + 16*q[2]**2*(-q[0]**2 - q[1]**2 - q[2]**2 + 1) - 4*q[2]**2 + 8*q[2]*cp.sqrt(-q[0]**2 - q[1]**2 - q[2]**2 + 1) - 3

# Create the problem
problem = cp.Problem(cp.Maximize(objective))
print("Is DCP:", objective.is_dcp())
print("Is DCP:", problem.is_dcp())
try:
	problem.solve(solver=cp.ECOS)
except cp.DCPError as e:
	print(e)
exit()
# Solve the problem
problem.solve(solver=cp.ECOS)


# Access the results
optimal_q = q.value
optimal_objective_value = problem.value

print(optimal_q)
print(optimal_objective_value)