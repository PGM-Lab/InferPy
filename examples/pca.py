import edward as ed
import inferpy as inf
from inferpy.models import Normal

K = 5
d = 10
N = 200

with inf.ProbModel() as m:
    with inf.replicate(size=K):
        mu = Normal(0, 1, dim=d)                    # Shape [K,d]

    sigma = 1.0
    mu0 = Normal(0, 1, dim=d)                       # Shape [1,d]

    with inf.replicate(size=N):
        w = Normal(0,1,dim=K)
        x = Normal(mu0 + inf.matmul(w,mu), sigma, observed=True)


    x_train = inf.models.Normal(loc=0, scale=1., dim=10).sample(N)
    data = {x.name: x_train}

    m.compile()
    m.fit(data)
    m.posterior([mu, mu0])
