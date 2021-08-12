import numpy as np
from scipy import sparse
import scipy.sparse.linalg as splinalg
from scipy.sparse.linalg import cg, LinearOperator, spsolve, spilu
import torch
import time
from scipy import stats
from scipy.sparse import spdiags, issparse, random, eye

## TODO : random density
def generate_Axb(count=100, n=1000, m=1, to_torch=True):
    As = []
    Bs = []
    xs = []
    rng = np.random.default_rng()
    rvs = stats.poisson(25, loc=10).rvs
    for i in range(count):
        A = random(n,n, density=0.05, random_state=rng, data_rvs=rvs)
        A = A + A.transpose() + n*eye(n)
        B = np.random.rand(n, )
        x = spsolve(A, B)
        # print(type(A))
        A = np.expand_dims(A.todense(), axis=0)
        # print(A.shape)
        As.append(A)
        Bs.append(B)
        xs.append(x)
        # check_close = np.allclose(A.dot(x), B)
        # print(check_close, np.linalg.norm(A.dot(x)-B))
    # print(As[0].shape)
    # print(len(As))
    As = np.stack(As)
    Bs = np.stack(Bs)
    xs = np.stack(xs)
    # print(As.shape, Bs.shape, xs.shape)
    if to_torch:
        As = torch.from_numpy(As)
        Bs = torch.from_numpy(Bs)
        xs = torch.from_numpy(xs)
    return As, Bs, xs

As, Bs, xs = generate_Axb(count=5, n=512, to_torch=False)
print(type(As))