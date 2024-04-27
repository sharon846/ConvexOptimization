from sympy import symbols, solve, I
import numpy as np

def filter_sol(solutions):
	real_sols = []
	solutions = np.array(solutions).astype(np.complex128)

	for sol in solutions:
		if np.isclose(np.abs(sol), 1.0):
			real_sols.append(sol)
			
	return np.array(real_sols)

def solve_eq(A, B, C, D, E):
	x = symbols('x')
	equation = A * x ** 4 + B * x ** 3 + C * x ** 2 + D * x + E

	solutions = solve(equation, x)
	return solutions

def solver(mat_A, vec_B):
	a1, a2, a3, a4 = mat_A.flatten()
	b1, b2 = vec_B.flatten()
	
	A = 0.5 * (a1 * a2 + a3 * a4)
	B = (a2 ** 2 - a1 ** 2 + a4 ** 2 - a3 ** 2) / (4 * I)
	C = 0.5 * (a2 * b1 + a4 * b2) 
	D = (a1 * b1 + a3 * b2) / (2 * I)
	
	solutions = solve_eq(A + B, D - C, 0, -D - C, A - B)
	valid_solutions = filter_sol(solutions)
	return valid_solutions



A = np.array([[3, -1], [2, 1]])
B = np.array([2, -1])

extreme_points = solver(A, B)
print(len(extreme_points))
print(extreme_points)

