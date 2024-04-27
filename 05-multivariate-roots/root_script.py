import numpy as np
import math

import warnings

# Filter out the specific warning
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in sqrt")

class IntervalSearchObject(object):
	def __init__(self, n_features, cost_func, depth=6, get_min=True):
		self.n_features = n_features
		self.cost = cost_func
		self.min = get_min

		self.MAX_BOUND = 1
		self.MIN_BOUND = -1

		self.max_depth = depth
		self.__reset__()

	def __reset__(self):
		self.depth = 0
		self.center = [0.5 * (self.MAX_BOUND + self.MIN_BOUND)] * self.n_features
		self.radius = self.MAX_BOUND - self.center[0]

	def get_search_cube(self, units):
		ranges = [np.linspace(max(self.MIN_BOUND, coord - self.radius), min(self.MAX_BOUND, coord + self.radius), units) for coord in self.center]
		grids = np.meshgrid(*ranges, indexing='ij')
		points = np.stack(grids, axis=-1)
		return points

	def search(self, units=20):
		points = self.get_search_cube(units)

		vals = np.apply_along_axis(self.cost, axis=-1, arr=points)

		if self.min:
			vals[np.isnan(vals)] = np.inf
			idx = np.unravel_index(np.argmin(vals), vals.shape)
		else:
			vals[np.isnan(vals)] = -np.inf
			idx = np.unravel_index(np.argmax(vals), vals.shape)

		self.center = points[idx].reshape(-1)
		print(self.center, self.cost(self.center))
		self.depth += 1

		if self.depth < self.max_depth:
			self.radius = np.exp(-self.depth)
			self.search(units)


if __name__ == '__main__':
	
	A = np.array([[-1, 3], [-2, 1]])
	b = np.array([2, 1])


	cost = lambda x: np.linalg.norm(np.dot(A, x) - b)

	search_obj_min = IntervalSearchObject(len(b), cost, get_min = True)
	search_obj_max = IntervalSearchObject(len(b), cost, get_min = False)

	search_obj_min.search(units=20)
	search_obj_max.search(units=20)

	print(f'max is {search_obj_max.center}')
	print(f'min is {search_obj_min.center}')
