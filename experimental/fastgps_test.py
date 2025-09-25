#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 09:37:58 2025

@author: sam
"""

import fastgps
import qmcpy as qp
import numpy as np
import torch
import pandas as pd
from matplotlib import pyplot


device = "cpu"
if device!="mps":
    torch.set_default_dtype(torch.float64)
    
#colors = ["xkcd:"+color[:-1] for color in pd.read_csv("../../../xkcd_colors.txt",comment="#").iloc[:,0].tolist()][::-1]
_alpha = 0.25
WIDTH = 2*(500/72)
LINEWIDTH = 3
MARKERSIZE = 100

d = 1 
f_smooth = lambda x: 15*(x[:,0]-1/2)**2*torch.sin(2*torch.pi*x[:,0])
f = f_smooth
def fp(x):
    xg = x.clone().requires_grad_(True)
    yg = f(xg)
    yp = torch.autograd.grad(yg,xg,grad_outputs=torch.ones_like(yg))[0]
    return yp[:,0].detach()

xticks = torch.linspace(0,1,252,device=device)[1:-1,None]
yticks = f(xticks)
ypticks = fp(xticks)
print("xticks.shape = %s"%str(tuple(xticks.shape)))
print("yticks.shape = %s"%str(tuple(yticks.shape)))
print("ypticks.shape = %s"%str(tuple(ypticks.shape)))

gps = [
    fastgps.StandardGP(qp.KernelGaussian(1,torchify=True,device=device),seqs=qp.DigitalNetB2(1,seed=11,randomize="DS")),
    fastgps.FastGPDigitalNetB2(qp.KernelDigShiftInvar(1,torchify=True,device=device),seqs=qp.DigitalNetB2(1,seed=7,randomize="DS"),alpha=4),
    fastgps.FastGPLattice(qp.KernelShiftInvar(1,torchify=True,device=device),seqs=qp.Lattice(1,seed=7),alpha=4),
]

gps_grad = [
    fastgps.StandardGP(
        qp.KernelMultiTaskDerivs(qp.KernelGaussian(1,torchify=True,device=device),num_tasks=2),
        seqs = [qp.DigitalNetB2(1,seed=7,randomize="DS"),qp.DigitalNetB2(1,seed=11,randomize="DS")],
        derivatives = [torch.tensor([0],device=device),torch.tensor([1],device=device)],
    ),
    fastgps.FastGPDigitalNetB2(
        qp.KernelMultiTaskDerivs(qp.KernelDigShiftInvar(1,torchify=True,alpha=4,device=device),num_tasks=2),
        seqs = [qp.DigitalNetB2(1,seed=7,randomize="DS"),qp.DigitalNetB2(1,seed=11,randomize="DS")],
        derivatives = [torch.tensor([0],device=device),torch.tensor([1],device=device)],
    ),
    fastgps.FastGPLattice(
        qp.KernelMultiTaskDerivs(qp.KernelShiftInvar(1,torchify=True,alpha=4,device=device),num_tasks=2),
        seqs = [qp.Lattice(1,seed=7),qp.Lattice(1,seed=11)],
        derivatives = [torch.tensor([0],device=device),torch.tensor([1],device=device)],
    ),
]

pmeans = [None]*len(gps)
pci_lows = [None]*len(gps)
pci_highs = [None]*len(gps)
for i,gp in enumerate(gps):
    print(type(gp).__name__)
    x_next = gp.get_x_next(n=2**2)
    gp.add_y_next(f(x_next))
    gp.fit()
    pmeans[i],_,_,pci_lows[i],pci_highs[i] = gp.post_ci(xticks,confidence=0.95)
    print("\tl2 relative error = %.1e"%(torch.linalg.norm(yticks-pmeans[i])/torch.linalg.norm(yticks)))
    
    
pmeans_grad = [None]*len(gps_grad)
pci_lows_grad = [None]*len(gps_grad)
pci_highs_grad = [None]*len(gps_grad)
for i,gp in enumerate(gps_grad):
    print(type(gp).__name__)
    x_next = gp.get_x_next(n=[2**2,2**2])
    gp.add_y_next([f(x_next[0]),fp(x_next[1])])
    gp.fit()
    pmeans_grad[i],_,_,pci_lows_grad[i],pci_highs_grad[i] = gp.post_ci(xticks,confidence=0.95)
    print("\tl2 relative error = %.1e"%(torch.linalg.norm(yticks-pmeans_grad[i][0])/torch.linalg.norm(yticks)))
    
    
fig,ax = pyplot.subplots(nrows=len(gps),ncols=3,sharex=True,sharey=False,figsize=(WIDTH*1.5,WIDTH/len(gps)*3))
ax = ax.reshape((len(gps),3))
for i,gp in enumerate(gps):
    ax[i,0].set_ylabel(type(gp).__name__,fontsize="xx-large")
    ax[i,0].plot(xticks[:,0].cpu(),yticks.cpu(),color="k",linewidth=LINEWIDTH)
    ax[i,0].scatter(gp.x[:,0].cpu(),gp.y.cpu(),color="k",s=MARKERSIZE)
    ax[i,0].plot(xticks[:,0].cpu(),pmeans[i].cpu(),linewidth=LINEWIDTH)
    ax[i,0].fill_between(xticks[:,0].cpu(),pci_lows[i].cpu(),pci_highs[i].cpu(),alpha=_alpha)
for i,gp in enumerate(gps_grad):
    ax[i,1].plot(xticks[:,0].cpu(),yticks.cpu(),color="k",linewidth=LINEWIDTH)
    ax[i,2].plot(xticks[:,0].cpu(),ypticks.cpu(),color="k",linewidth=LINEWIDTH)
    ax[i,1].scatter(gp.x[0][:,0].cpu(),gp.y[0].cpu(),color="k",s=MARKERSIZE)
    ax[i,2].scatter(gp.x[1][:,0].cpu(),gp.y[1].cpu(),color="k",s=MARKERSIZE)
    ax[i,1].plot(xticks[:,0].cpu(),pmeans_grad[i][0].cpu(),linewidth=LINEWIDTH)
    ax[i,2].plot(xticks[:,0].cpu(),pmeans_grad[i][1].cpu(),linewidth=LINEWIDTH)
    ax[i,1].fill_between(xticks[:,0].cpu(),pci_lows_grad[i][0].cpu(),pci_highs_grad[i][0].cpu(),alpha=_alpha)
    ax[i,2].fill_between(xticks[:,0].cpu(),pci_lows_grad[i][1].cpu(),pci_highs_grad[i][1].cpu(),alpha=_alpha)
ax[0,0].set_title(r"$f$ no grad",fontsize="xx-large")
ax[0,1].set_title(r"$f$ with grad",fontsize="xx-large")
ax[0,2].set_title(r"$\mathrm{d} f / \mathrm{d} x$ with grad",fontsize="xx-large")
for j in range(3):
    ax[-1,j].set_xlabel(r"$x$",fontsize="xx-large")
# fig.savefig("./gps_deriv.pdf",bbox_inches="tight")