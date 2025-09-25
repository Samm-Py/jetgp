import fastgps
import qmcpy as qp
import torch
import numpy as np

device = "cpu"
if device!="mps":
    torch.set_default_dtype(torch.float64)
    
# def f_ackley(x, a=20, b=0.2, c=2*np.pi, scaling=32.768):
#     # https://www.sfu.ca/~ssurjano/ackley.html
#     assert x.ndim==2
#     x = 2*scaling*x-scaling
#     t1 = a*torch.exp(-b*torch.sqrt(torch.mean(x**2,1)))
#     t2 = torch.exp(torch.mean(torch.cos(c*x),1))
#     t3 = a+np.exp(1)
#     y = -t1-t2+t3
#     return y
# f_low_fidelity = lambda x: f_ackley(x,c=0)
# f_high_fidelity = lambda x: f_ackley(x)
# f_cos = lambda x: torch.cos(2*np.pi*x).sum(1)
# fs = [f_low_fidelity,f_high_fidelity,f_cos]
# d = 1 # dimension
# rng = torch.Generator().manual_seed(17)
# x = torch.rand((2**7,d),generator=rng).to(device) # random testing locations
# y = torch.vstack([f(x) for f in fs]) # true values at random testing locations
# z = torch.rand((2**8,d),generator=rng).to(device) # other random locations at which to evaluate covariance
# print("x.shape = %s"%str(tuple(x.shape)))
# print("y.shape = %s"%str(tuple(y.shape)))
# print("z.shape = %s"%str(tuple(z.shape)))

# fgp = fastgps.FastGPLattice(
#     qp.KernelMultiTask(
#         qp.KernelShiftInvar(d,torchify=True,device=device),
#         num_tasks=len(fs)),
#     seqs=7)
# x_next = fgp.get_x_next(n=[2**6,2**3,2**8])
# y_next = [fs[i](x_next[i]) for i in range(fgp.num_tasks)]
# fgp.add_y_next(y_next)
# assert len(x_next)==len(y_next)
# for i in range(len(x_next)):
#     print("i = %d"%i)
#     print("\tx_next[%d].shape = %s"%(i,str(tuple(x_next[i].shape))))
#     print("\ty_next[%d].shape = %s"%(i,str(tuple(y_next[i].shape))))
    
    
# pmean = fgp.post_mean(x)
# print("pmean.shape = %s"%str(tuple(pmean.shape)))
# print("l2 relative error =",(torch.linalg.norm(y-pmean,dim=1)/torch.linalg.norm(y,dim=1)))

# data = fgp.fit()
# list(data.keys())

from qmcpy import Lattice, fftbr, ifftbr
n = 8
d = 1
lat = Lattice(d,seed=11)
x = lat(n)
x.shape


kernel = qp.KernelShiftInvar(
d = d, 
alpha = list(range(2,d+2)),
scale = 1,
lengthscales = [1/j**2 for j in range(1,d+1)])

kmat = kernel(x[:,None,:],x[None,:,:])
