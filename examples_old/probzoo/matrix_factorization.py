import inferpy as inf
from inferpy.models import Normal

N=200
M=50
K=5

# Shape [M,K]
with inf.replicate(size=K):
    gamma = Normal(0,1, dim=M)

# Shape [N,K]
with inf.replicate(size=N):
    w = Normal(0,1, dim=K)

# x_mn has shape [N,K] x [K,M] = [N,M]

with inf.replicate(size=N):
    x = Normal(inf.matmul(w,gamma), 1, observed = True)

m = inf.ProbModel([w,gamma,x])

data = m.sample(size=N)

log_prob = m.log_prob(data)

m.compile(infMethod = 'KLqp')

m.fit(data)

print(m.posterior([w,gamma]))