import unittest

import inferpy as inf
import numpy as np


class operators_test(unittest.TestCase):
    def test(self):
        sess = inf.util.Runtime.tf_sess

        theta = inf.models.Normal(loc=[50], scale=0.000001, name="theta")
        x = inf.models.Normal(loc=[theta, 2 * theta, theta * -4], scale=0.000001, name="x", dim=3)

        th = 0.001

        op1 = (x + 100).sample()
        op2 = x.sample() + 100
        self.assertTrue(np.all(abs(op1 - op2) < th))

        op1 = (x * 3).sample()
        op2 = x.sample() * 3
        self.assertTrue(np.all(abs(op1 - op2) < th))

        op1 = (x - 100).sample()
        op2 = x.sample() - 100
        self.assertTrue(np.all(abs(op1 - op2) < th))

        op1 = (x / 3).sample()
        op2 = x.sample() / 3
        self.assertTrue(np.all(abs(op1 - op2) < th))

        # review this op
        #op1 = (x // 2).sample()
        #op2 = x.sample() // 2
        #self.assertTrue(np.all(abs(op1 - op2[0]) < th))

        # review this operation
        op1 = (x % 3).sample()
        op2 = x.sample() % 3
        self.assertTrue(np.all(abs(op1 - op2) < th))

        op1 = (x < 2).sample()
        op2 = x.sample() < 2
        self.assertTrue(np.all(op1 == op2))

        op1 = (x > 2).sample()
        op2 = x.sample() > 2
        self.assertTrue(np.all(op1 == op2))

        op1 = (x <= 2).sample()
        op2 = x.sample() <= 2
        self.assertTrue(np.all(op1 == op2))

        #		op1 = (x[1])
        #		op2 = x.sample()[0, 1]
        #		self.assertTrue(np.all(abs(op1 - op2) < th))

        op1 = (x ** 2).sample()
        op2 = x.sample() ** 2
        self.assertTrue(np.all(abs(op1 - op2) < th * 10))

        op1 = (-x).sample()
        op2 = -x.sample()
        self.assertTrue(np.all(abs(op1 - op2) < th))

        op1 = (abs(x)).sample()
        op2 = abs(x.sample())
        self.assertTrue(np.all(abs(op1 - op2) < th))


        op1 = ((x>0) & (x>80)).sample()
        op2 = (x.sample()>0) & (x.sample()>80)
        self.assertTrue(np.all(op1 == op2))

        op1 = ((x>0) | (x>80)).sample()
        op2 = (x.sample()>0) | (x.sample()>80)
        self.assertTrue(np.all(op1 == op2))



        op1 = (x>0).equal(x>1).sample()
        op2 = (x.sample()>0)==(x.sample()>1)
        self.assertTrue(np.all(op1 == op2))




if __name__ == '__main__':
    unittest.main()


