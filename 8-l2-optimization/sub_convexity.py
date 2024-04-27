import numpy as np
import cvxpy as cp
from itertools import product

def build_objective(A, b):
	objective = ""
	components = []
	
	for r in range(A.shape[0]):
		components.append("")
		for c in range(A.shape[1]):
			components[-1] += str(A[r,c]) + "*x[" + str(c) + "]+"
		
		components[-1] = "(" + components[-1][:-1] + "-" + str(b[r]) + ")"

	objective = " ** 2 + ".join(components)
	objective +=  " ** 2"
	
	return objective, components

def solve_subproblem(components, function, indicators):

	d = len(components)

	x = cp.Variable(d)
	
	constraints = [cp.norm(x, 2) <= 1]
	for _ in range(d):
		constraints.append(eval(components[_]) * indicators[_] >= 0)

	objective = cp.Minimize(eval(function))
	problem = cp.Problem(objective, constraints)
	problem.solve(solver=cp.CVXOPT)

	if (np.isinf(problem.value)):
		return None
		
	return problem.value, x.value


A = np.array([[3, -1, 1], [2, 1, 2], [2, 5, -1]])
b = np.array([2, -1, 3])

objective, components = build_objective(A,b)
# print(objective)

slices = list(product([1, -1], repeat=len(components)))
slices = [list(comb) for comb in slices]


coordinate = None
global_min = np.inf

for slice in slices:

	res = solve_subproblem(components, objective, slice)
	if res is None:
		continue
		
	value, point = res
	if value < global_min:
		global_min = value
		coordinate = point

print(f"Global minimum is {global_min} at point {coordinate} that has norm of {np.linalg.norm(coordinate)}")


