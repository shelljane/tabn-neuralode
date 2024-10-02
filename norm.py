import sys
sys.path.append(".")
import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from einops import rearrange, repeat

import torchdiffeq
from torchdiffeq import odeint_adjoint
from botorch.models import SingleTaskGP, ModelListGP
from botorch.models.transforms import Normalize, Standardize

import gpytorch
import botorch
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood, ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_model

from torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline

import warnings
warnings.filterwarnings("ignore")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHID = 0
def setEpoch(epoch): 
    global EPOCHID
    EPOCHID = epoch


class NoNorm(nn.Identity): 
    def __init__(self, dim): 
        super().__init__()

    def forward(self, t, x): 
        return super().forward(x)
    

class BatchNorm1d(nn.BatchNorm1d): 
    def __init__(self, dim): 
        super().__init__(dim, track_running_stats=False)

    def forward(self, t, x): 
        return super().forward(x)
    

class BatchNorm2d(nn.BatchNorm2d): 
    def __init__(self, dim): 
        super().__init__(dim, track_running_stats=False)

    def forward(self, t, x): 
        return super().forward(x)
    

class BadNorm2d(nn.BatchNorm2d): 
    def __init__(self, dim): 
        super().__init__(dim, track_running_stats=True)

    def forward(self, t, x): 
        return super().forward(x)
    

class BadNorm1d(nn.BatchNorm1d): 
    def __init__(self, dim): 
        super().__init__(dim, track_running_stats=True)

    def forward(self, t, x): 
        return super().forward(x)


class LayerNorm(nn.LayerNorm): 
    def __init__(self, dim): 
        super().__init__(dim)

    def forward(self, t, x): 
        return super().forward(x)


class InstanceNorm2d(nn.InstanceNorm2d): 
    def __init__(self, dim): 
        super().__init__(dim)

    def forward(self, t, x): 
        return super().forward(x)


class BatchNorm2dT0(nn.Module): 
    def __init__(self, dim): 
        super().__init__()
        self._dim = dim
        self._mean = torch.zeros(self._dim, dtype=torch.float32, device=DEVICE)
        self._var = torch.zeros(self._dim, dtype=torch.float32, device=DEVICE)
        self._bnweight = nn.Parameter(torch.ones(self._dim, dtype=torch.float32, device=DEVICE), requires_grad=True)
        self._bnbias = nn.Parameter(torch.zeros(self._dim, dtype=torch.float32, device=DEVICE), requires_grad=True)
        self._init = False

    def forward(self, t, x): 
        if not self._init: 
            self._mean = torch.mean(x, dim=[idx for idx in range(len(x.shape)) if idx != 1])
            self._var = torch.var(x, dim=[idx for idx in range(len(x.shape)) if idx != 1])
            self._init = True
        return torch.nn.functional.batch_norm(x, self._mean, self._var, self._bnweight, self._bnbias, training=(self.training and abs(t.item()) <= 1e-3))


class BatchNorm2dEuler(nn.Module): 
    def __init__(self, dim, timesteps=100, timerange=[0, 1], rescale=True): 
        super().__init__()
        self._dim = dim
        self._weights = [nn.Parameter(torch.ones(self._dim, dtype=torch.float32, device=DEVICE), requires_grad=True) for _ in range(timesteps)]
        self._biases = [nn.Parameter(torch.zeros(self._dim, dtype=torch.float32, device=DEVICE), requires_grad=True) for _ in range(timesteps)]
        self._means = [torch.zeros(self._dim, dtype=torch.float32, device=DEVICE) for _ in range(timesteps)]
        self._vars = [torch.ones(self._dim, dtype=torch.float32, device=DEVICE) for _ in range(timesteps)]
        self._step = (timerange[1] - timerange[0]) / timesteps
        self._times = np.arange(timerange[0], timerange[1], step=self._step)
        self._inits = [False for _ in range(timesteps)]
        self._momentum = 0.1
        self._rescale = rescale
        for idx, weight in enumerate(self._weights): 
            self.register_parameter(f"bn-weight{idx}", weight)
        for idx, bias in enumerate(self._biases): 
            self.register_parameter(f"bn-bias{idx}", bias)
        self._count = 0

    def forward(self, t, x): 
        t = t.item()
        index = np.argmin(np.abs(self._times - t))

        mean = self._means[index]
        var = self._vars[index]

        weight = None
        bias = None
        if self._rescale: 
            weight = self._weights[index]
            bias = self._biases[index]
        
        return F.batch_norm(x, mean, var, weight, bias, training=self.training, momentum=self._momentum)


def argTopK(array, top_k=1):
    partitioned = np.argpartition(array, kth=top_k)[:top_k]
    selected = array[partitioned]
    sortedIdx = np.argsort(selected)
    return partitioned[sortedIdx]
class BatchNorm2dBi(nn.Module): 
    def __init__(self, dim, timesteps=100, timerange=[0, 1], rescale=True): 
        super().__init__()
        self._dim = dim
        self._weights = [nn.Parameter(torch.ones(self._dim, dtype=torch.float32, device=DEVICE), requires_grad=True) for _ in range(timesteps)]
        self._biases = [nn.Parameter(torch.zeros(self._dim, dtype=torch.float32, device=DEVICE), requires_grad=True) for _ in range(timesteps)]
        self._means = [torch.zeros(self._dim, dtype=torch.float32, device=DEVICE) for _ in range(timesteps)]
        self._vars = [torch.ones(self._dim, dtype=torch.float32, device=DEVICE) for _ in range(timesteps)]
        self._step = (timerange[1] - timerange[0]) / timesteps
        self._times = np.arange(timerange[0], timerange[1], step=self._step)
        self._inits = [False for _ in range(timesteps)]
        self._momentum = 0.1
        self._rescale = rescale
        for idx, weight in enumerate(self._weights): 
            self.register_parameter(f"bnweight{idx}", weight)
        for idx, bias in enumerate(self._biases): 
            self.register_parameter(f"bnbias{idx}", bias)
        self._count = 0

    def forward(self, t, x): 
        t = t.item()
        only = False
        if t <= self._times[0] or t >= self._times[-1]: 
            only = True
        if not only: 
            dists = t - self._times
            sortedIdx = argTopK(np.abs(dists), top_k=2)
            if dists[sortedIdx[0]] >= 0: 
                index1 = sortedIdx[0]
                index2 = sortedIdx[1]
            else: 
                index1 = sortedIdx[1]
                index2 = sortedIdx[0]
            t1 = self._times[index1]
            t2 = self._times[index2]
            step1 = (t - t1) / (t2 - t1)
            step2 = (t2 - t) / (t2 - t1)
        elif t <= self._times[0]: 
            index = 0
        elif t >= self._times[-1]: 
            index = -1
        else: 
            assert 0, "Illegal time for batch normalization"

        if self.training: 
            self._count += 1
            mean = torch.mean(x, dim=[idx for idx in range(len(x.shape)) if idx != 1]).detach()
            var = torch.var(x, dim=[idx for idx in range(len(x.shape)) if idx != 1]).detach()
            if not only: 
                if not self._inits[index1]: 
                    # self._means[index1] = mean
                    # self._vars[index1] = var
                    self._inits[index1] = True
                else: 
                    momentum = self._momentum * step1
                    self._means[index1] = (1-momentum) * self._means[index1] + momentum * mean
                    self._vars[index1] = (1-momentum) * self._vars[index1] + momentum * var
                if not self._inits[index2]: 
                    # self._means[index2] = mean
                    # self._vars[index2] = var
                    self._inits[index2] = True
                else: 
                    momentum = self._momentum * step2
                    self._means[index2] = (1-momentum) * self._means[index2] + momentum * mean
                    self._vars[index2] = (1-momentum) * self._vars[index2] + momentum * var
            else: 
                if not self._inits[index]: 
                    # self._means[index] = mean
                    # self._vars[index] = var
                    self._inits[index] = True
                else: 
                    momentum = self._momentum
                    self._means[index] = (1-momentum) * self._means[index] + momentum * mean
                    self._vars[index] = (1-momentum) * self._vars[index] + momentum * var

        if not only: 
            mean = step2 * self._means[index1] + step1 * self._means[index2]
            var = step2 * self._vars[index1] + step1 * self._vars[index2]
        else: 
            mean = self._means[index]
            var = self._vars[index]

        weight = None
        bias = None
        if self._rescale: 
            if not only: 
                weight = step2 * self._weights[index1] + step1 * self._weights[index2]
                bias = step2 * self._biases[index1] + step1 * self._biases[index2]
            else: 
                weight = self._weights[index]
                bias = self._biases[index]
        
        return F.batch_norm(x, mean, var, weight, bias, training=self.training, momentum=0)
class BatchNorm2dBi(nn.Module): 
    def __init__(self, dim, timesteps=100, timerange=[0, 1], rescale=True): 
        super().__init__()
        self._dim = dim
        self._weights = [nn.Parameter(torch.ones(self._dim, dtype=torch.float32, device=DEVICE), requires_grad=True) for _ in range(timesteps)]
        self._biases = [nn.Parameter(torch.zeros(self._dim, dtype=torch.float32, device=DEVICE), requires_grad=True) for _ in range(timesteps)]
        self._means = [torch.zeros(self._dim, dtype=torch.float32, device=DEVICE) for _ in range(timesteps)]
        self._vars = [torch.ones(self._dim, dtype=torch.float32, device=DEVICE) for _ in range(timesteps)]
        self._step = (timerange[1] - timerange[0]) / timesteps
        self._times = np.arange(timerange[0], timerange[1], step=self._step)
        self._inits = [False for _ in range(timesteps)]
        self._momentum = 0.1
        self._rescale = rescale
        for idx, weight in enumerate(self._weights): 
            self.register_parameter(f"bnweight{idx}", weight)
        for idx, bias in enumerate(self._biases): 
            self.register_parameter(f"bnbias{idx}", bias)
        self._count = 0

    def forward(self, t, x): 
        t = t.item()
        only = False
        if t <= self._times[0] or t >= self._times[-1]: 
            only = True
        if not only: 
            dists = t - self._times
            sortedIdx = argTopK(np.abs(dists), top_k=2)
            if dists[sortedIdx[0]] >= 0: 
                index1 = sortedIdx[0]
                index2 = sortedIdx[1]
            else: 
                index1 = sortedIdx[1]
                index2 = sortedIdx[0]
            t1 = self._times[index1]
            t2 = self._times[index2]
            step1 = (t - t1) / (t2 - t1)
            step2 = (t2 - t) / (t2 - t1)
        elif t <= self._times[0]: 
            index = 0
        elif t >= self._times[-1]: 
            index = -1
        else: 
            assert 0, "Illegal time for batch normalization"

        if self.training: 
            self._count += 1
            mean = torch.mean(x, dim=[idx for idx in range(len(x.shape)) if idx != 1]).detach()
            var = torch.var(x, dim=[idx for idx in range(len(x.shape)) if idx != 1]).detach()
            if not only: 
                if not self._inits[index1]: 
                    # self._means[index1] = mean
                    # self._vars[index1] = var
                    self._inits[index1] = True
                else: 
                    momentum = self._momentum * step1
                    self._means[index1] = (1-momentum) * self._means[index1] + momentum * mean
                    self._vars[index1] = (1-momentum) * self._vars[index1] + momentum * var
                if not self._inits[index2]: 
                    # self._means[index2] = mean
                    # self._vars[index2] = var
                    self._inits[index2] = True
                else: 
                    momentum = self._momentum * step2
                    self._means[index2] = (1-momentum) * self._means[index2] + momentum * mean
                    self._vars[index2] = (1-momentum) * self._vars[index2] + momentum * var
            else: 
                if not self._inits[index]: 
                    # self._means[index] = mean
                    # self._vars[index] = var
                    self._inits[index] = True
                else: 
                    momentum = self._momentum
                    self._means[index] = (1-momentum) * self._means[index] + momentum * mean
                    self._vars[index] = (1-momentum) * self._vars[index] + momentum * var

        if not only: 
            mean = step1 * self._means[index1] + step2 * self._means[index2]
            var = step1 * self._vars[index1] + step2 * self._vars[index2]
        else: 
            mean = self._means[index]
            var = self._vars[index]

        weight = None
        bias = None
        if self._rescale: 
            if not only: 
                weight = step1 * self._weights[index1] + step2 * self._weights[index2]
                bias = step1 * self._biases[index1] + step2 * self._biases[index2]
            else: 
                weight = self._weights[index]
                bias = self._biases[index]
        
        return F.batch_norm(x, mean, var, weight, bias, training=self.training, momentum=0)
    

class BatchNorm2dSmooth(nn.Module): 
    def __init__(self, dim, timesteps=100, timerange=[0, 1], rescale=True): 
        super().__init__()
        self._dim = dim
        self._weights = nn.Parameter(torch.ones((timesteps, self._dim), dtype=torch.float32, device=DEVICE), requires_grad=True)
        self._biases = nn.Parameter(torch.zeros((timesteps, self._dim), dtype=torch.float32, device=DEVICE), requires_grad=True)
        self._means = torch.zeros((timesteps, self._dim), dtype=torch.float32, device=DEVICE)
        self._vars = torch.ones((timesteps, self._dim), dtype=torch.float32, device=DEVICE)
        self._timesteps = timesteps
        self._step = (timerange[1] - timerange[0]) / timesteps
        self._times = np.arange(timerange[0], timerange[1], step=self._step)
        self._times = torch.tensor(self._times, dtype=torch.float32, device=DEVICE)
        self._inits = [False for _ in range(timesteps)]
        self._momentum = 1.0
        # self._momentum = 0.2
        self._rescale = rescale
        self._count = 0

    def weights(self, t): 
        window = round(self._timesteps/10) * self._step
        time = min(max(self._times[0], t), self._times[-1])
        weights = 1/math.sqrt(2*math.pi) * torch.exp(-0.5 * ((self._times - time)/window)**2)
        weights = weights / weights.sum()
        return weights

    def forward(self, t, x): 
        t = t.item()
        scales = self.weights(t)

        if self.training: 
            if t >= self._times[0] and t <= self._times[-1]: 
                self._count += 1
                mean = torch.mean(x, dim=[idx for idx in range(len(x.shape)) if idx != 1]).detach()
                var = torch.var(x, dim=[idx for idx in range(len(x.shape)) if idx != 1]).detach()
                global EPOCHID
                # momentum = max(self._momentum * 0.98**EPOCHID, 0.2)
                momentum = min(self._momentum * 1.01**EPOCHID, 2.0)
                momentum = torch.minimum(momentum * scales, torch.ones_like(scales)).view(-1, 1)
                self._means = (1-momentum) * self._means + momentum * mean
                self._vars = (1-momentum) * self._vars + momentum * var
                # for idx in range(len(self._times)): 
                #     if scales[idx].item() < 1e-3: 
                #         continue
                #     momentum = self._momentum * scales[idx]
                #     self._means[idx] = (1-momentum) * self._means[idx] + momentum * mean
                #     self._vars[idx] = (1-momentum) * self._vars[idx] + momentum * var
            
                mean = None
                var = None
            else: 
                mean = None
                var = None
        else: 
            mean = scales.view(1, -1) @ self._means
            var = scales.view(1, -1) @ self._vars

        weight = None
        bias = None
        if self._rescale: 
            weight = scales.view(1, -1) @ self._weights
            bias = scales.view(1, -1) @ self._biases
        
        return F.batch_norm(x, mean, var, weight, bias, training=self.training, momentum=0)


class BatchNorm2dCubic(nn.Module): 
    def __init__(self, dim, timesteps=10, timerange=[0, 1], rescale=True): 
        super().__init__()
        self._dim = dim
        self._counts = [0 for _ in range(timesteps)]
        self._bnweights = nn.Parameter(torch.ones([timesteps, self._dim], dtype=torch.float32, device=DEVICE), requires_grad=True)
        self._bnbiases = nn.Parameter(torch.zeros([timesteps, self._dim], dtype=torch.float32, device=DEVICE), requires_grad=True)
        self._means = torch.zeros([timesteps, self._dim], dtype=torch.float32, device=DEVICE).requires_grad_(True)
        self._vars = torch.ones([timesteps, self._dim], dtype=torch.float32, device=DEVICE).requires_grad_(True)
        self._step = (timerange[1] - timerange[0]) / timesteps
        self._times = torch.linspace(timerange[0], timerange[1], timesteps, dtype=torch.float32, device=DEVICE)
        self._inits = [False for _ in range(timesteps)]
        self._momentum = 0.1
        self._rescale = rescale
        self._count = 0
        self._opt = optim.AdamW([self._means, self._vars], lr=1e-3)

    def forward(self, t, x): 
        coeffsMeans = natural_cubic_spline_coeffs(self._times, self._means)
        splineMeans = NaturalCubicSpline(coeffsMeans)
        coeffsVars = natural_cubic_spline_coeffs(self._times, self._vars)
        splineVars = NaturalCubicSpline(coeffsVars)
        mean = splineMeans.evaluate(t).detach()
        var = splineVars.evaluate(t).detach()
        if self._rescale: 
            coeffsWeights = natural_cubic_spline_coeffs(self._times, self._bnweights)
            splineWeights = NaturalCubicSpline(coeffsWeights)
            coeffsBiases = natural_cubic_spline_coeffs(self._times, self._bnbiases)
            splineBiases = NaturalCubicSpline(coeffsBiases)
            weight = splineWeights.evaluate(t)
            bias = splineBiases.evaluate(t)
        else: 
            weight = None
            bias = None

        result = F.batch_norm(x, mean, var, weight, bias, training=self.training, momentum=0)
        
        mean = torch.mean(x, dim=[idx for idx in range(len(x.shape)) if idx != 1]).detach()
        var = torch.var(x, dim=[idx for idx in range(len(x.shape)) if idx != 1]).detach()
        mean2 = splineMeans.evaluate(t)
        var2 = splineVars.evaluate(t)
        if self.training and mean2.requires_grad and var2.requires_grad: 
            for idx in range(4): 
                coeffsMeans = natural_cubic_spline_coeffs(self._times, self._means)
                splineMeans = NaturalCubicSpline(coeffsMeans)
                coeffsVars = natural_cubic_spline_coeffs(self._times, self._vars)
                splineVars = NaturalCubicSpline(coeffsVars)
                mean2 = splineMeans.evaluate(t)
                var2 = splineVars.evaluate(t)
                loss = F.mse_loss(mean2, mean) + F.mse_loss(var2, var)
                loss.backward()
                nn.utils.clip_grad_norm_([self._means, self._vars], 0.1)
                self._opt.step()
                self._opt.zero_grad()

        return result


class BatchNorm2dGP(nn.Module): 
    def __init__(self, dim, timesteps=10, interval=100, timerange=[0, 1], rescale=True, cubic=False, bilinear=False): 
        super().__init__()
        self._dim = dim
        self._timesteps = timesteps
        self._interval = interval
        self._counts = [0 for _ in range(timesteps)]
        self._bnweights = nn.Parameter(torch.ones([timesteps, self._dim], dtype=torch.float32, device=DEVICE), requires_grad=True)
        self._bnbiases = nn.Parameter(torch.zeros([timesteps, self._dim], dtype=torch.float32, device=DEVICE), requires_grad=True)
        self._means = [torch.zeros(self._dim, dtype=torch.float32, device=DEVICE) for _ in range(timesteps)]
        self._vars = [torch.ones(self._dim, dtype=torch.float32, device=DEVICE) for _ in range(timesteps)]
        self._step = (timerange[1] - timerange[0]) / timesteps
        self._times = np.arange(timerange[0], timerange[1], step=self._step)
        self._inits = [False for _ in range(timesteps)]
        self._momentum = 0.1
        self._rescale = rescale
        self._count = 0
        self._cachedTime = []
        self._cachedMean = []
        self._cachedVar = []
        self._modelMean = None
        self._modelVar = None
        self._modelWeight = None
        self._modelBias = None
        self._countTrained = 0
        self._cubic = cubic
        self._bilinear = bilinear
        if self._cubic: 
            self._coeffsWeights = natural_cubic_spline_coeffs(torch.tensor(self._times, dtype=torch.float32, device=DEVICE), self._bnweights)
            self._splineWeights = NaturalCubicSpline(self._coeffsWeights)
            self._coeffsBiases = natural_cubic_spline_coeffs(torch.tensor(self._times, dtype=torch.float32, device=DEVICE), self._bnbiases)
            self._splineBiases = NaturalCubicSpline(self._coeffsBiases)
        self._init = False

    def forward(self, t, x): 
        t = t.item()

        if self.training and (not self._init or len(self._cachedTime) >= self._interval): 
            with torch.enable_grad():
                trainX = torch.tensor(self._times, dtype=torch.float32, device=DEVICE).view(-1, 1)
                trainY = self._bnweights
                trainY.requires_grad_(False)
                model = SingleTaskGP(train_X=trainX, train_Y=trainY, covar_module=gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()), input_transform=Normalize(d=trainX.shape[-1]), outcome_transform=Standardize(m=trainY.shape[-1])).to(DEVICE)
                fit_gpytorch_model(ExactMarginalLogLikelihood(model.likelihood, model))
                self._modelWeight = model
                trainY.requires_grad_(True)
                
                trainX = torch.tensor(self._times, dtype=torch.float32, device=DEVICE).view(-1, 1)
                trainY = self._bnbiases
                trainY.requires_grad_(False)
                model = SingleTaskGP(train_X=trainX, train_Y=trainY, covar_module=gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()), input_transform=Normalize(d=trainX.shape[-1]), outcome_transform=Standardize(m=trainY.shape[-1])).to(DEVICE)
                fit_gpytorch_model(ExactMarginalLogLikelihood(model.likelihood, model))
                trainY.requires_grad_(True)
                self._modelBias = model
            self._init = True

        if len(self._cachedTime) >= self._interval: 
            with torch.enable_grad():
                trainX = torch.tensor(self._cachedTime, dtype=torch.float32, device=DEVICE).view(-1, 1)
                trainY = torch.stack(self._cachedMean, dim=0)
                model = SingleTaskGP(train_X=trainX, train_Y=trainY, covar_module=gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()), input_transform=Normalize(d=trainX.shape[-1]), outcome_transform=Standardize(m=trainY.shape[-1])).to(DEVICE)
                fit_gpytorch_model(ExactMarginalLogLikelihood(model.likelihood, model))
                self._modelMean = model
                predX = torch.tensor(self._times, dtype=torch.float32, device=DEVICE).view(-1, 1)
                predY = self._modelMean.posterior(predX)
                update = predY.mean.detach()
                momentum = max(self._momentum, 1/(self._countTrained+1))
                for idx in range(len(self._means)): 
                    self._means[idx] = (1 - momentum) * self._means[idx] + momentum * update[idx]
                    
                trainX = torch.tensor(self._cachedTime, dtype=torch.float32, device=DEVICE).view(-1, 1)
                trainY = torch.stack(self._cachedVar, dim=0)
                model = SingleTaskGP(train_X=trainX, train_Y=trainY, covar_module=gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()), input_transform=Normalize(d=trainX.shape[-1]), outcome_transform=Standardize(m=trainY.shape[-1])).to(DEVICE)
                fit_gpytorch_model(ExactMarginalLogLikelihood(model.likelihood, model))
                self._modelVar = model
                predX = torch.tensor(self._times, dtype=torch.float32, device=DEVICE).view(-1, 1)
                predY = self._modelVar.posterior(predX)
                update = predY.mean.detach()
                momentum = max(self._momentum, 1/(self._countTrained+1))
                for idx in range(len(self._vars)): 
                    self._vars[idx] = (1 - momentum) * self._vars[idx] + momentum * update[idx]
            self._cachedTime = []
            self._cachedMean = []
            self._cachedVar = []
            self._countTrained += 1
            # self._interval = min(round(self._interval * 1.05), 500)

        if self.training: 
            self._count += 1
            mean = torch.mean(x, dim=[idx for idx in range(len(x.shape)) if idx != 1]).detach()
            var = torch.var(x, dim=[idx for idx in range(len(x.shape)) if idx != 1]).detach()
            self._cachedTime.append(t)
            self._cachedMean.append(mean)
            self._cachedVar.append(var)

        predX = torch.tensor(t, dtype=torch.float32, device=DEVICE).view(1, 1)
        mean = self._modelMean.posterior(predX).mean[0].detach() if not self.training and not self._modelMean is None else None
        var = self._modelVar.posterior(predX).mean[0].detach() if not self.training and not self._modelVar is None else None

        weight = None
        bias = None
        if self._rescale: 
            if self._bilinear: 
                only = False
                if t <= self._times[0] or t >= self._times[-1]: 
                    only = True
                if not only: 
                    dists = t - self._times
                    sortedIdx = argTopK(np.abs(dists), top_k=2)
                    if dists[sortedIdx[0]] >= 0: 
                        index1 = sortedIdx[0]
                        index2 = sortedIdx[1]
                    else: 
                        index1 = sortedIdx[1]
                        index2 = sortedIdx[0]
                    t1 = self._times[index1]
                    t2 = self._times[index2]
                    step1 = (t - t1) / (t2 - t1)
                    step2 = (t2 - t) / (t2 - t1)
                elif t <= self._times[0]: 
                    index = 0
                elif t >= self._times[-1]: 
                    index = -1
                else: 
                    assert 0, "Illegal time for batch normalization"
                if not only: 
                    weight = step1 * self._bnweights[index1] + step2 * self._bnweights[index2]
                    bias = step1 * self._bnbiases[index1] + step2 * self._bnbiases[index2]
                else: 
                    weight = self._bnweights[index]
                    bias = self._bnbiases[index]
            elif self._cubic: 
                weight = self._splineWeights.evaluate(torch.tensor(t, dtype=torch.float32, device=DEVICE))
                bias = self._splineBiases.evaluate(torch.tensor(t, dtype=torch.float32, device=DEVICE))
            else: 
                predX = torch.tensor(t, dtype=torch.float32, device=DEVICE).view(1, 1)
                weight = self._modelWeight.posterior(predX).mean[0]
                bias = self._modelBias.posterior(predX).mean[0]
        
        return F.batch_norm(x, mean, var, weight, bias, training=self.training, momentum=0)


def normlayer(dim, ntype="batchnorm"): 
    result = None
    if ntype == "batchnorm": 
        result = BatchNorm2d(dim)
    elif ntype == "nonorm": 
        result = NoNorm(dim)
    elif ntype == "badnorm": 
        result = BadNorm2d(dim)
    elif ntype == "badnorm1d": 
        result = BadNorm1d(dim)
    elif ntype == "batchnorm1d": 
        result = BatchNorm1d(dim)
    elif ntype == "layernorm":
        result = LayerNorm(dim)
    elif ntype == "instnorm":
        result = InstanceNorm2d(dim)
    elif ntype == "euler":
        result = BatchNorm2dEuler(dim)
    elif ntype == "bilinear":
        result = BatchNorm2dBi(dim)
    elif ntype == "smooth":
        result = BatchNorm2dSmooth(dim)
    elif ntype == "cubic":
        result = BatchNorm2dCubic(dim)
    elif ntype == "gpsimple":
        result = BatchNorm2dGP(dim, rescale=False)
    elif ntype == "gpbilinear":
        result = BatchNorm2dGP(dim, bilinear=True)
    elif ntype == "gpcubic":
        result = BatchNorm2dGP(dim, cubic=True)
    elif ntype == "gaussian":
        result = BatchNorm2dGP(dim)
    return result
