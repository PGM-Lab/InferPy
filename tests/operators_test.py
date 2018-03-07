import unittest

import inferpy as inf
import numpy as np


class operators_test(unittest.TestCase):
	def test(self):
		sess = inf.util.Runtime.tf_sess

		theta = inf.models.Normal(loc=[50], scale=0.000001, name="theta")
		x = inf.models.Normal(loc=[theta, 2 * theta, theta * -4], scale=0.000001, name="x", dim=3)

		th = 0.001

		op1 = sess.run(x + 100)
		op2 = x.sample() + 100
		self.assertTrue(np.all(abs(op1 - op2) < th))

		op1 = sess.run(x * 3)
		op2 = x.sample() * 3
		self.assertTrue(np.all(abs(op1 - op2) < th))

		op1 = sess.run(x - 100)
		op2 = x.sample() - 100
		self.assertTrue(np.all(abs(op1 - op2) < th))

		op1 = sess.run(x / 3)
		op2 = x.sample() / 3
		self.assertTrue(np.all(abs(op1 - op2) < th))

		op1 = sess.run(x // 2)
		op2 = x.sample() // 2
		self.assertTrue(np.all(abs(op1 - op2[0]) < th))

		# review this operation
		op1 = sess.run(x % 3)
		op2 = x.sample() % 3
		self.assertTrue(np.all(abs(op1 - op2) < th))

		op1 = sess.run(x < 2)
		op2 = x.sample() < 2
		self.assertTrue(np.all(op1 == op2))

		op1 = sess.run(x > 2)
		op2 = x.sample() > 2
		self.assertTrue(np.all(op1 == op2))

		op1 = sess.run(x <= 2)
		op2 = x.sample() <= 2
		self.assertTrue(np.all(op1 == op2))

		op1 = sess.run(x[1])
		op2 = x.sample()[0, 1]
		self.assertTrue(np.all(abs(op1 - op2) < th))

		op1 = sess.run(x ** 2)
		op2 = x.sample() ** 2
		self.assertTrue(np.all(abs(op1 - op2) < th * 10))

		op1 = sess.run(-x)
		op2 = -x.sample()
		self.assertTrue(np.all(abs(op1 - op2) < th))

		op1 = sess.run(abs(x))
		op2 = abs(x.sample())
		self.assertTrue(np.all(abs(op1 - op2) < th))


if __name__ == '__main__':
	unittest.main()


