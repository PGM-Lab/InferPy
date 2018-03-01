import unittest
import edward as ed


reload(ed)
ed.set_seed(1234)

import inferpy as inf


class param_inference_test(unittest.TestCase):
    def test(self):

        #### learning a 1-dim parameter from 1-dim data

        N = 50
        sampling_mean = [30.]
        sess = ed.util.get_session()

        with inf.ProbModel() as m:
            theta = inf.models.Normal(loc=0., scale=1.)

            with inf.replicate(size=N):
                x = inf.models.Normal(loc=theta, scale=1., observed=True)

        m.compile()

        x_train = inf.models.Normal(loc=sampling_mean, scale=1.).sample(N)
        data = {x.name: x_train}

        m.fit(data)

        print(m.posterior(theta).loc[0])
        self.assertTrue(abs(m.posterior(theta).loc[0]-29.017122)<0.000001)



        #### learning a 2-dim parameter from 2-dim data


        N = 50
        sampling_mean = [30., 10.]
        sess = ed.util.get_session()

        with inf.ProbModel() as m:
            theta = inf.models.Normal(loc=0., scale=1., dim=2)

            with inf.replicate(size=N):
                x = inf.models.Normal(loc=theta, scale=1., observed=True)

        m.compile()

        x_train = inf.models.Normal(loc=sampling_mean, scale=1.).sample(N)
        data = {x.name: x_train}

        m.fit(data)

        print(m.posterior(theta).loc[0])
        print(m.posterior(theta).loc[1])


        self.assertTrue(abs(m.posterior(theta).loc[0]-28.963005)<0.000001)
        self.assertTrue(abs(m.posterior(theta).loc[1]-9.735327)<0.000001)


        #### learning two 1-dim parameter from 2-dim data


        N = 50
        sampling_mean = [30., 10.]
        sess = ed.util.get_session()

        with inf.ProbModel() as m:
            theta1 = inf.models.Normal(loc=0., scale=1., dim=1)
            theta2 = inf.models.Normal(loc=0., scale=1., dim=1)

            with inf.replicate(size=N):
                x = inf.models.Normal(loc=[theta1, theta2], scale=1., observed=True)

        m.compile()

        x_train = inf.models.Normal(loc=sampling_mean, scale=1.).sample(N)
        data = {x.name: x_train}

        m.fit(data)


        print(m.posterior(theta1).loc[0])
        print(m.posterior(theta2).loc[0])

        self.assertTrue(abs(m.posterior(theta1).loc[0]-29.10924)<0.000001)
        self.assertTrue(abs(m.posterior(theta2).loc[0]-9.936194)<0.000001)




if __name__ == '__main__':
    unittest.main()


