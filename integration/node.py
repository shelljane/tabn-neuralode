import sys
sys.path.append(".")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from einops import rearrange, repeat
import torchdiffeq
from torchdiffeq import odeint_adjoint

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from resnet import *

class dfwrapper(nn.Module):
    def __init__(self, df, shape, recf=None):
        super(dfwrapper, self).__init__()
        self.df = df
        self.shape = shape
        self.recf = recf

    def forward(self, t, x):
        bsize = x.shape[0]
        if self.recf:
            x = x[:, :-self.recf.osize].reshape(bsize, *self.shape)
            dx = self.df(t, x)
            dr = self.recf(t, x, dx).reshape(bsize, -1)
            dx = dx.reshape(bsize, -1)
            dx = torch.cat([dx, dr], dim=1)
        else:
            x = x.reshape(bsize, *self.shape)
            dx = self.df(t, x)
            dx = dx.reshape(bsize, -1)
        return dx


ODE_METHOD = "dopri5"
# ODE_METHOD = "euler"
TIME_STEP = 1e-2
class NODEintegrate(nn.Module):

    def __init__(self, df, shape=None, tol=1e-3, adjoint=True, evaluation_times=None, recf=None):
        """
        Create an OdeRnnBase model
            x' = df(x)
            x(t0) = x0
        :param df: a function that computes derivative. input & output shape [batch, channel, feature]
        :param x0: initial condition.
            - if x0 is set to be nn.parameter then it can be trained.
            - if x0 is set to be nn.Module then it can be computed through some network.
        """
        super().__init__()
        self.df = dfwrapper(df, shape, recf) if shape else df
        self.tol = tol
        self.odeint = torchdiffeq.odeint_adjoint if adjoint else torchdiffeq.odeint
        self.evaluation_times = evaluation_times if evaluation_times is not None else torch.tensor([0.0, 1.0], device=DEVICE)
        self.shape = shape
        self.recf = recf
        if recf:
            assert shape is not None
        self.method = ODE_METHOD

    def forward(self, x0):
        """
        Evaluate odefunc at given evaluation time
        :param x0: shape [batch, channel, feature]. Set to None while training.
        :param evaluation_times: time stamps where method evaluates, shape [time]
        :param x0stats: statistics to compute x0 when self.x0 is a nn.Module, shape required by self.x0
        :return: prediction by ode at evaluation_times, shape [time, batch, channel, feature]
        """
        bsize = x0.shape[0]
        if self.shape:
            assert x0.shape[1:] == torch.Size(self.shape), \
                'Input shape {} does not match with model shape {}'.format(x0.shape[1:], self.shape)
            x0 = x0.reshape(bsize, -1)
            if self.recf:
                reczeros = torch.zeros_like(x0[:, :1])
                reczeros = repeat(reczeros, 'b 1 -> b c', c=self.recf.osize)
                x0 = torch.cat([x0, reczeros], dim=1)
            try: 
                out = odeint_adjoint(self.df, x0, self.evaluation_times, rtol=self.tol, atol=self.tol, method=self.method, options={"step_size": TIME_STEP, "max_num_steps": 2**31-1})
            except: 
                print(f"RK45 failed, roll back to euler method with stepsize=1e-2")
                out = odeint_adjoint(self.df, x0, self.evaluation_times, rtol=self.tol, atol=self.tol, method="euler", options={"step_size": TIME_STEP, "max_num_steps": 2**31-1})
            # print(f" ->> {x0.shape} {out.shape}")
            if self.recf:
                rec = out[-1, :, -self.recf.osize:]
                out = out[:, :, :-self.recf.osize]
                out = out.reshape(-1, bsize, *self.shape)
                return out, rec
            else:
                return out
        else:
            try: 
                out = odeint_adjoint(self.df, x0, self.evaluation_times, rtol=self.tol, atol=self.tol, method=self.method, options={"step_size": TIME_STEP, "max_num_steps": 2**31-1})
            except: 
                print(f"RK45 failed, roll back to euler method with stepsize=1e-2")
                out = odeint_adjoint(self.df, x0, self.evaluation_times, rtol=self.tol, atol=self.tol, method="euler", options={"step_size": TIME_STEP, "max_num_steps": 2**31-1})
            # print(f" --> {x0.shape} {out.shape}")
            return out

    @property
    def nfe(self):
        return self.df.nfe

    def to(self, device, *args, **kwargs):
        super().to(device, *args, **kwargs)
        self.evaluation_times.to(device)


class NODElayer(NODEintegrate):
    def forward(self, x0):
        out = super(NODElayer, self).forward(x0)
        if isinstance(out, tuple):
            out, rec = out
            return out[-1], rec
        else:
            return out[-1]


class NODE(nn.Module):
    def __init__(self, df=None, **kwargs):
        super(NODE, self).__init__()
        self.__dict__.update(kwargs)
        self.df = df
        self.dim = df.dim
        self.nfe = 0
        self.elem_t = None

    def forward(self, t, x):
        self.nfe += 1
        if self.elem_t is None:
            return self.df(t, x)
        else:
            return self.elem_t * self.df(self.elem_t, x)

    def update(self, elem_t):
        self.elem_t = elem_t.view(*elem_t.shape, 1)


class NODEinit(nn.Module): 
    def __init__(self, chOut, aug=1): 
        super().__init__()
        self.aug = aug
        self.chOut = chOut
    def forward(self, x0): 
        outshape = list(x0.shape)
        outshape[1] = self.aug * self.chOut
        out = torch.zeros(outshape).to(DEVICE)
        out[:, :x0.shape[1]] += x0
        out = rearrange(out, 'b (d c) ... -> b d c ...', c=self.chOut)
        return out


class NODEwritein(nn.Module): 
    def __init__(self): 
        super().__init__()
    def forward(self, x0): 
        out = rearrange(x0, 'b c x y -> b 1 c x y')
        return out


class NODEreadout(nn.Module): 
    def __init__(self,): 
        super().__init__()

    def forward(self, x0): 
        return rearrange(x0, 'b d c x y -> b (d c) x y')


class NODEbatchnorm(nn.Module): 
    def __init__(self, kernel): 
        super().__init__()
        self.bn = nn.BatchNorm2d(kernel)

    def forward(self, x0): 
        x0 = rearrange(x0, 'b d c x y -> b (d c) x y')
        x = self.bn(x0)
        out = rearrange(x, 'b c x y -> b 1 c x y')
        return out


