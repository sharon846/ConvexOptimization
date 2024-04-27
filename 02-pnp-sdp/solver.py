import cvxpy as cp
import numpy as np


# Set a random seed for reproducibility
# Set a random seed for reproducibility
np.random.seed(42)

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

def f(x):
	return np.array([
			x[0]**2+x[1]**2-x[2]**2-x[3]**2, 2*x[1]*x[2]-2*x[0]*x[3], 2*x[1]*x[3]+2*x[0]*x[2],\
			2*x[1]*x[2]+2*x[0]*x[3], x[0]**2+x[2]**2-x[1]**2-x[3]**2, 2*x[2]*x[3]-2*x[0]*x[1],\
			2*x[1]*x[3]-2*x[0]*x[2], 2*x[2]*x[3]+2*x[0]*x[1], x[0]**2+x[3]**2-x[1]**2-x[2]**2])


def is_rotation_matrix(matrix):
    if not isinstance(matrix, np.ndarray) or matrix.shape != (3, 3):
        return False
    
    # Compute the transpose of the matrix
    transpose_matrix = np.transpose(matrix)
    
    # Compute the inverse of the matrix
    inverse_matrix = np.linalg.inv(matrix)
    
    # Check if the transpose equals the inverse
    return np.allclose(transpose_matrix, inverse_matrix)

# Generate a random 9x9 matrix A and a random vector b
A, B = generate_positive_semidefinite_matrix(9)

# Define optimization variable
# x = cp.Variable(4)
X = cp.Variable((9, 9), PSD=True)

# Create the problem
problem = cp.Problem(cp.Minimize(cp.trace(B @ X)), [cp.trace(X) == 1, X >>0])

print("Is DCP:", problem.is_dcp())
# Solve the problem
problem.solve('CVXOPT')

print("Trace is " + str(np.trace(X.value)))
print("PSD IS " + str(is_positive_semidefinite(X.value)))

U, D, _ = np.linalg.svd(X.value)
optimal_q = U[:, 0] * np.sqrt(D[0])

print("Norm of solution is: " +  str(np.linalg.norm(optimal_q)))
print("Is rotation matrix: " +  str(is_rotation_matrix(optimal_q.reshape(3,3))))