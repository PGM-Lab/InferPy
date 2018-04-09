import unittest

import inferpy as inf
import numpy as np


class Normal_test_definition(unittest.TestCase):
	def test(self):

		loc = 5.
		scale = [0.2, 0.2, 0.2]
		dim = 3
		N = 100

		x = inf.models.Normal(loc=loc, scale=scale, dim=dim)
		sample_x = x.sample(N)
		shape_x = np.shape(sample_x)
		mean_x = np.mean(sample_x)


		print(shape_x)
		self.assertTrue(shape_x == (100,3))
		self.assertTrue(abs(mean_x - np.mean(loc)) < np.max(scale))



if __name__ == '__main__':
	unittest.main()


