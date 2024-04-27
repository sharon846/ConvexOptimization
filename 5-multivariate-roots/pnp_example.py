import numpy as np
import math
from sympy import symbols, expand, simplify, nan
from root_script import *


# A = np.array([[-1, 3], [-2, 1]])
# B = np.array([2, 1])

# def min_norm2_on_unit_vec(x):
	# a1 = (A[0,0] * x + A[0,1] * np.sqrt(1-x**2) - B[0]) ** 2 + (A[1,0] * x + A[1,1] * np.sqrt(1-x**2) - B[1]) ** 2
	# a2 = (A[0,0] * x - A[0,1] * np.sqrt(1-x**2) - B[0]) ** 2 + (A[1,0] * x - A[1,1] * np.sqrt(1-x**2) - B[1]) ** 2

	# a = np.row_stack((a1, a2))

	# return np.min(a, axis=0)

# search_obj_min = IntervalSearchObject(len(B), min_norm2_on_unit_vec, get_min = True)
# search_obj_min.search(units=20)
# print(f'min is {search_obj_min.center} and f(x)={min_norm2_on_unit_vec(search_obj_min.center)}')

def pnp_example(point):
	x1, y1, z1 = point

	a1 = polynom.subs({x: x1, y: y1, z: z1, w: np.sqrt(1-x1**2-y1**2-z1**2)})
	a2 = polynom.subs({x: x1, y: y1, z: z1, w: -np.sqrt(1-x1**2-y1**2-z1**2)})
	
	minimum_value = min(a1, a2)
	
	if np.isnan(minimum_value):
		return np.nan
		
	else:
		return int(minimum_value)


def get_expr(idx):
	if idx == 0:
		return "x**2+y**2-z**2-(1-x**2-y**2-z**2)"
	if idx == 1:
		return "2*y*z-2*x*w"
	if idx == 2:
		return "2*x*z+2*y*w"
	if idx == 3:
		return "2*y*z+2*x*w"
	if idx == 4:
		return "x**2+z**2-y**2-(1-x**2-y**2-z**2)"
	if idx == 5:
		return "-2*x*y+2*z*w"
	if idx == 6:
		return "-2*x*z+2*y*w"
	if idx == 7:
		return "2*x*y+2*z*w"
	if idx == 8:
		return "x**2+(1-x**2-y**2-z**2)-y**2-z**2"

def real_f_q(B):
	sum = ""
	for i in range(9):
		for j in range(9):
			sum += str(B[i,j]) + "*(" + get_expr(i) + ")*(" + get_expr(j) + ")"
			sum += "+"
	sum = sum[:-1]
	return sum


Mr = np.array([[1.0137, 0.0012, 0.1757, 0.2151, -0.0037, 0.0155, -0.0247, 0.0022, 0.0095],
               [0.0012, 0.0677, 0.1323, -0.1415, 0.0017, -0.0126, 0.0034, -0.0024, -0.0018],
               [0.1757, 0.1323, 0.6310, -0.4504, 0.0052, -0.0405, 0.0087, 0.0012, 0.0249],
               [0.2151, -0.1415, -0.4504, 0.4900, -0.0440, -0.0446, -0.0130, 0.0022, -0.0092],
               [-0.0037, 0.0017, 0.0052, -0.0440, 0.1008, 0.2386, -0.0025, 0.0014, 0.0061],
               [0.0155, -0.0126, -0.0405, -0.0446, 0.2386, 0.9636, -0.0118, 0.0065, 0.0224],
               [-0.0247, 0.0034, 0.0087, -0.0130, -0.0025, -0.0118, 0.0110, -0.0003, 0.0004],
               [0.0022, -0.0024, 0.0012, 0.0022, 0.0014, 0.0065, -0.0003, 0.0009, 0.0021],
               [0.0095, -0.0018, 0.0249, -0.0092, 0.0061, 0.0224, 0.0004, 0.0021, 0.0085]])


Mr = 1.0e+03 * Mr

x, y, z, w = symbols('x y z w')
polynom = real_f_q(Mr)
polynom = expand(polynom)


search_obj_min = IntervalSearchObject(3, pnp_example, depth=40, get_min = True)
search_obj_min.search(units=20)

point = search_obj_min.center
val = pnp_example(point)
# point = validate_solution(point)
print(f'min is at {point} and f(x)={val}')
