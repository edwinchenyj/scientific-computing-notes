import networkx as nx
from scipy import sparse
import scipy.sparse.linalg as splinalg
import numpy as np
import torch
from cg_batch import CG, cg_batch
# import IPython as ipy
import time
from scipy import stats
from scipy.sparse import spdiags, issparse, random, eye

torch.set_default_tensor_type(torch.DoubleTensor)

def sparse_numpy_to_torch(A):
    rows, cols = A.nonzero()
    values = A.data
    indices = np.vstack((rows, cols))
    i = torch.LongTensor(indices)
    v = torch.DoubleTensor(values)
    return torch.sparse.DoubleTensor(i, v, A.shape)

n = 1000
m = 1
K = 8

## number of systems to solve
# K = 16
As_torch = [None] * K
Ms_torch = [None] * K
# B_torch = [None] * K
B_torch = torch.DoubleTensor(K, n, m)
print(B_torch.requires_grad)
rng = np.random.default_rng()
rvs = stats.poisson(25, loc=10).rvs
n = 1000
for i in range(K):
    A = random(n,n, density=0.05, random_state=rng, data_rvs=rvs)
    A = A + A.transpose() + n*eye(n)
    b = np.random.rand(n, m)
    M_column = np.sqrt(np.diag(A.todense()))
    M = spdiags(M_column,0,n,n).tocsc()
    As_torch[i] = sparse_numpy_to_torch(A)
    Ms_torch[i] = sparse_numpy_to_torch(M)
    B_torch[i] = torch.tensor(b, requires_grad=True)
    # print(B_torch[i].requires_grad)

def A_bmm(X):
    Y = [(As_torch[i]@X[i]).unsqueeze(0) for i in range(K)]
    return torch.cat(Y, dim=0)

def M_bmm(X):
    Y = [(Ms_torch[i]@X[i]).unsqueeze(0) for i in range(K)]
    return torch.cat(Y, dim=0)

# def A_bmm_2(X):
#     Y = A_bdiag_torch@(X.view(K * n, m))
#     return Y.view(K, n, m)


# def M_bmm_2(X):
#     Y = M_bdiag_torch@(X.view(K * n, m))
#     return Y.view(K, n, m)

# print(f"Solving K={K} linear systems that are {n} x {n} with {As[0].nnz} nonzeros and {m} right hand sides.")

## Use the solver directly w/o considering network issue
X, _ = cg_batch(A_bmm, B_torch, M_bmm=M_bmm, rtol=1e-6, atol=0.0, maxiter=100, verbose=True)

## TODO : fix the network issue -> pytorch static thing
# cg = CG(A_bmm, M_bmm=M_bmm, rtol=1e-5, atol=1e-5, verbose=True)
# X = cg(B_torch)

# start = time.perf_counter()
# X_np = np.concatenate([np.hstack([splinalg.cg(A, B[:, i], M=M)[0][:, np.newaxis] for i in range(m)])[np.newaxis, :, :]
#                        for A, B, M in zip(As, Bs, Ms)], 0)
# end = time.perf_counter()
# print("Scipy took %.3f seconds" % (end - start))
# np.testing.assert_allclose(X_np, X.cpu().data.numpy(), atol=1e-4)