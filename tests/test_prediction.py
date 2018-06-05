import unittest
import inferpy as inf
import edward as ed
import numpy as np

class Prediction_test(unittest.TestCase):
    def test(self):
        # import the package for using it



        sess = ed.get_session()

        flag_y = True
        f = 2
        # graph



        N = 1000  # number of observations
        # model definition
        with inf.ProbModel() as m:
            # prior (latent variable)
            beta = inf.models.Normal(loc=0, scale=1, name="beta")
            w = inf.models.Normal(loc=0, scale=1, name="w")
            b = inf.models.Normal(loc=0, scale=1, name="b")
            betaz = inf.models.Normal(loc=0, scale=1, name="beta")

            # observed variable
            with inf.replicate(size=N):
                z = inf.models.Normal(loc=betaz, scale=1, observed=False, name="z")
                x = inf.models.Normal(loc=beta + z, scale=1, observed=True, name="x")
                y = inf.models.Normal(loc=w * x + b + z, scale=1, observed=True, name="y")

        # toy data generation

        x_train = inf.models.Normal(loc=10, scale=3, dim=1).sample(N)
        y_train = x_train * f + inf.models.Normal(loc=1, scale=0.1, dim=1).sample(N)

        data = {x.name: x_train, y.name: y_train}

        m.compile()
        m.fit(data)

        qbeta = m.posterior(beta)
        qw = m.posterior(w)
        qb = m.posterior(b)
        qz = m.posterior(z)

        x_test = inf.models.Normal(loc=10, scale=3, dim=1).sample(N)
        y_test = x_test * f + inf.models.Normal(loc=1, scale=0.1, dim=1).sample(N)

        y_pred = m.predict(y, data={x: x_test}).loc

        self.assertTrue(np.max((y_pred - y_test) < 0.5))

        import inferpy.criticism.evaluate as idc


        
        idc.evaluate("mean_squared_error", y_pred, data={x:x_test})




if __name__ == '__main__':
    unittest.main()


