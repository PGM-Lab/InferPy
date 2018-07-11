import unittest
import inferpy as inf
import numpy as np
import tensorflow as tf



class Test_get_items(unittest.TestCase):
    def test(self):
        d = 100
        N = 1

        L = [i for i in range(0, d * N)]

        with inf.replicate(size=N):
            n = inf.models.Categorical(probs=[0.0, 1.0], dim=1)
            x = inf.models.Normal(L, 0.000001)

        self.assertTrue(np.all(abs(x[5].loc - 5) < 0.01))
        self.assertTrue(np.all(abs(x[n].loc - 1) < 0.01))

        ########



        d = 1
        N = 100

        L = [[i] for i in range(0, d * N)] * inf.models.Normal(1, 0.00001)

        with inf.replicate(size=N):
            n = inf.models.Categorical(probs=[1.0, 0.0], dim=1)
            x = inf.models.Normal(L, 0.000001, dim=1)



        self.assertTrue(x[0].shape == [N, 1])
        self.assertTrue(x[10, 0].shape == [1, 1])

        self.assertTrue(np.all(abs(x[5, 0].loc - 5) < 0.01))

        ########



        d = 10
        N = 10

        L = tf.constant(np.reshape([i for i in range(0, d * N)], (N, d)), dtype="float32")

        with inf.replicate(size=N):
            n = inf.models.Categorical(probs=[1.0, 0.0], dim=1)
            x = inf.models.Normal(L, 0.000001, dim=d)

        self.assertTrue(x[0].shape == [N, 1])
        self.assertTrue(x[10, 0].shape == [1, 1])

        self.assertTrue(np.all(abs(x[5, 0].loc - 50) < 0.01))
        self.assertTrue(np.all(abs((x[n])[3, 0].loc - 30) < 0.01))


if __name__ == '__main__':
    unittest.main()


