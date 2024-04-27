import numpy as np
import cvxpy as cp
from itertools import product
from sympy import symbols, solve, Abs

def build_objective(A, b):
	objective = "cp.abs("
	components = []
	
	for r in range(len(b)):
		components.append("")
		for c in range(1,len(b)+1):
			components[-1] += str(A[r,c-1]) + f"*x{c}+"
		
		components[-1] = components[-1][:-1] + "-" + str(b[r])

	objective += ") + cp.abs(".join(components)
	objective += ")"
	
	return objective, components

def solve_subproblem(components, function, idx):

	n = len(components)
	d = 3

	null_hyper_plane_value = solve(components[idx], f"x{d}")[0]
	# print(null_hyper_plane_value, "and", components[idx])
	# exit()
	
	new_objective = function.replace("cp.abs", "Abs")		# for sympy usage
	new_objective = eval(new_objective)
	new_objective = new_objective.subs({eval(f"x{d}"): null_hyper_plane_value})
	new_objective = str(new_objective).replace("Abs", "cp.abs")
	
	constraint = eval("x1 ** 2 + x2 ** 2 + x3 ** 2")
	constraint = constraint.subs({eval(f"x{d}"): null_hyper_plane_value})
	constraint = str(constraint)
	
	x1 = cp.Variable()
	x2 = cp.Variable()
	
	new_objective = cp.Minimize(eval(new_objective))
	problem = cp.Problem(new_objective, [eval(constraint) <= 1])
	
	# print(problem)
	#print(components[idx])
	problem.solve(solver=cp.CVXOPT)
	#print(problem.value, x1.value, x2.value, constraint)
	#input()
	
	if (np.isinf(problem.value)):
		return None
	
	# print(x1.value)

	return problem.value, [x1.value, x2.value]


A = np.array([[3, -1, 1], [2, 1, 2], [1, 2.5, -1]])
b = np.array([2, -1, 3])

x1, x2, x3 = symbols("x1 x2 x3")

objective, components = build_objective(A,b)

print(objective)

coordinate = None
global_min = np.inf

for idx in range(len(b)):

	res = solve_subproblem(components, objective, idx)
	if res is None:
		continue
		
	value, point = res
	if value < global_min:
		global_min = value
		coordinate = point

print(f"Global minimum is {global_min} at point {coordinate} that has norm of {np.linalg.norm(coordinate)}")


