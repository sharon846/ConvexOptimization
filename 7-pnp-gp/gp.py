import cvxpy as cp
import numpy as np


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

def f_q(x, y, z, w):
	# x, y, z, w = q
	return 1.4462036939465634*w**4 - 1.2748268818575219*w**3*x - 8.7398554914671269*w**3*y - 5.7112648970530286*w**3*z + 2.7050778961455419*w**2*x**2 - 0.14332017286013232*w**2*x*y + 1.188433450923349*w**2*x*z + 40.683096385019211*w**2*y**2 + 69.426137190613034*w**2*y*z + 41.360752374912523*w**2*z**2 - 0.39532005410109794*w*x**3 + 48.211212718179601*w*x**2*y + 50.57248739968642*w*x**2*z + 28.806234873811151*w*x*y**2 + 25.289859701404008*w*x*y*z + 2.0450095072060068*w*x*z**2 - 17.238227231173662*w*y**3 + 53.972907778853763*w*y**2*z + 36.286317453203049*w*y*z**2 - 27.144485754238348*w*z**3 + 16.656043389198661*x**4 + 15.520086083700006*x**3*y + 0.89276130479611071*x**3*z - 3.2316085987377142*x**2*y**2 + 45.763816242952184*x**2*y*z - 6.7454384727666981*x**2*z**2 - 4.961945364393218*x*y**3 + 23.641511277207374*x*y**2*z - 0.38290830112114504*x*y*z**2 - 0.13494025916679711*x*z**3 + 3.6908789424228169*y**4 - 18.784282338838397*y**3*z + 42.143098687804722*y**2*z**2 - 26.700710952098466*y*z**3 + 7.212982949514599*z**4
	
# Generate a random 9x9 matrix A and a random vector b
A, B = generate_positive_semidefinite_matrix(9)

# Define optimization variable
x = cp.Variable(pos=True)
y = cp.Variable(pos=True)
z = cp.Variable(pos=True)
w = cp.Variable(pos=True)

objective = cp.Maximize(f_q(x, y, z, w))

# Create the problem
problem = cp.Problem(objective)
problem.solve(gp=True)

# Access the results
optimal_q = q.value
optimal_objective_value = problem.value

print(optimal_q)
print(optimal_objective_value)