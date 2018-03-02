import unittest


import edward as ed
import tensorflow as tf
import numpy as np



import inferpy as inf





class param_inference_test(unittest.TestCase):
    def test(self):

        #### learning a 1-dim parameter from 1-dim data

        N = 500
        sampling_mean = [30.]
        sampling_std = 0.0000001
        sess = ed.util.get_session()

        with inf.ProbModel() as m:
            theta = inf.models.Normal(loc=0., scale=1.)

            with inf.replicate(size=N):
                x = inf.models.Normal(loc=theta, scale=1., observed=True)

        m.compile()

        x_train = inf.models.Normal(loc=sampling_mean, scale=sampling_std).sample(N)
        data = {x.name: x_train}

        m.fit(data)

        p1 = m.posterior(theta).loc[0]





        #### learning a 2-dim parameter from 2-dim data


        sampling_mean = [30., 10.]
        sess = ed.util.get_session()

        with inf.ProbModel() as m:
            theta = inf.models.Normal(loc=0., scale=1., dim=2)

            with inf.replicate(size=N):
                x = inf.models.Normal(loc=theta, scale=1., observed=True)

        m.compile()

        x_train = inf.models.Normal(loc=sampling_mean, scale=sampling_std).sample(N)
        data = {x.name: x_train}

        m.fit(data)

        p2_1 = m.posterior(theta).loc[0]
        p2_2 = m.posterior(theta).loc[1]






        #### learning two 1-dim parameter from 2-dim data




        sampling_mean = [30., 10.]
        sess = ed.util.get_session()

        with inf.ProbModel() as m:
            theta1 = inf.models.Normal(loc=0., scale=1., dim=1)
            theta2 = inf.models.Normal(loc=0., scale=1., dim=1)

            with inf.replicate(size=N):
                x = inf.models.Normal(loc=[theta1, theta2], scale=1., observed=True)

        m.compile()

        x_train = inf.models.Normal(loc=sampling_mean, scale=sampling_std).sample(N)
        data = {x.name: x_train}

        m.fit(data)

        p3_1 = m.posterior(theta1).loc[0]
        p3_2 = m.posterior(theta2).loc[0]


        ## asserts


        print(p1)
        print(p2_1)
        print(p2_2)
        print(p3_1)
        print(p3_2)

        self.assertTrue(abs(p1 - 29.57) < 0.1)


        self.assertTrue(abs(p2_1 - 29.57) < 0.1)
        self.assertTrue(abs(p2_2 - 9.98) < 0.1)

        self.assertTrue(abs(p3_1-29.57)<0.1)
        self.assertTrue(abs(p3_2-9.98)<0.1)



if __name__ == '__main__':
    unittest.main()


