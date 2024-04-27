import numpy as np
from sympy.parsing.sympy_parser import parse_expr
from sympy import symbols, expand, log, simplify, collect, factor, diff, solveset, Interval, Eq, solve

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

# Define the matrix Mr
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
# print(Mr)
# exit()

polynom = real_f_q(Mr)
polynom = str(expand(polynom))
# polynom_representation1 = polynom.replace('w', '\sqrt{1-x**2-y**2-z**2}')
# polynom_representation2 = polynom.replace('w', '(-\sqrt{1-x**2-y**2-z**2})')

print(polynom)
print(polynom_representation1.replace('**', "^").replace('*', '\cdot '))
print(polynom_representation2.replace('**', "^").replace('*', '\cdot '))
exit()


