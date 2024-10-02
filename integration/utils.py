import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint
import numpy as np
from einops import rearrange, repeat
import time
import torch.optim as optim
import glob
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from math import pi
from random import random
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Normal
import torchvision as tv
from torchvision import datasets, transforms
import argparse
import csv

import utils
import models

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# class ArgumentParser:
#     def add_argument(self, str, type, default):
#         setattr(self, str[2:], default)

#     def parse_args(self):
#         return self

def str_rec (names, data, unit=None, sep=', ', presets='{}'):
    if unit is None:
        unit = [''] * len(names)
    data = [str(i)[:6] for i in data]
    out_str = "{}: {{}} {{{{}}}}" + sep
    out_str *= len(names)
    out_str = out_str.format(*names)
    out_str = out_str.format(*data)
    out_str = out_str.format(*unit)
    out_str = presets.format(out_str)
    return out_str

def cifar(batch_size=64, size=32, path_to_data='../data'):
    """MNIST dataloader with (3, 28, 28) images.
    Parameters
    ----------
    batch_size : int
    size : int
        Size (height and width) of each image. Default is 28 for no resizing.
    path_to_data : string
        Path to MNIST data files.
    """
    all_transforms = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])

    train_data = datasets.CIFAR10(path_to_data, train=True, download=True,
                                transform=all_transforms)
    test_data = datasets.CIFAR10(path_to_data, train=False,
                               transform=all_transforms)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader

def cifar(batch_size=64, size=32, path_to_data='../data'):
    preprocTrain = tv.transforms.Compose([
            tv.transforms.ToTensor(), 
            tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
            tv.transforms.RandomHorizontalFlip(), 
            tv.transforms.RandomErasing(scale=(0.04, 0.2), ratio=(0.5, 2)), 
            tv.transforms.RandomCrop(32, padding=4)])
    preprocTest = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    trainset = tv.datasets.CIFAR10(root=path_to_data, train=True, download=True, transform=preprocTrain)
    testset = tv.datasets.CIFAR10(root=path_to_data, train=False, download=True, transform=preprocTest)
    loaderTrain = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    loaderTest = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
    return loaderTrain, loaderTest

from tqdm import tqdm
def train(model, optimizer, trdat, tsdat, args):
    rec_names = ["iter", "loss", "acc", "nfe", "forwardnfe", "time/iter", "time"]
    rec_unit = ["","","","","","s","min"]
    itrcnt = 0
    loss_func = nn.CrossEntropyLoss()
    itr_arr = np.zeros(args.niters)
    loss_arr = np.zeros(args.niters)
    nfe_arr = np.zeros(args.niters)
    forward_nfe_arr = np.zeros(args.niters)
    time_arr = np.zeros(args.niters)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.95)
    outlist = []
    # csvfile = open(f'../results/cifar/tol/{args.model}/{args.model}_{args.tol}_.csv', 'w')
    # writer = csv.writer(csvfile)
    # training

    bnweights = []
    bnbiases = []
    for name, param in model.named_parameters(): 
        if "bnweight" in name: 
            bnweights.append(param)
        if "bnbias" in name: 
            bnbiases.append(param)
    def bnloss(): 
        result = 0
        for weight in bnweights: 
            result += F.mse_loss(weight, torch.ones_like(weight))
        for bias in bnbiases: 
            result += F.mse_loss(bias, torch.zeros_like(bias))
        return result

    start_time = time.time()
    for epoch in range(1, args.niters+1):
        print(f"Epoch {epoch}")
        acc = 0
        dsize = 0
        iter_start_time = time.time()
        for x, y in tqdm(trdat):
            x = x.to(device=f'cuda:{args.gpu}')
            y = y.to(device=f'cuda:{args.gpu}')
            itrcnt += 1
            model[1].df.nfe = 0
            optimizer.zero_grad()
            # forward in time and solve ode
            pred_y = model(x)
            forward_nfe_arr[epoch - 1] += model[1].df.nfe
            # compute loss
            loss = loss_func(pred_y, y) + 0.1 * bnloss()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            # make arrays
            itr_arr[epoch - 1] = epoch
            loss_arr[epoch - 1] += loss.detach()
            nfe_arr[epoch - 1] += model[1].df.nfe
            # compute acc
            pred_l = torch.argmax(pred_y, dim=1)
            acc += torch.sum((pred_l == y).float())
            dsize += y.shape[0]
        iter_end_time = time.time()
        time_arr[epoch - 1] = iter_end_time - iter_start_time
        loss_arr[epoch - 1] *= 1.0 * epoch / itrcnt
        nfe_arr[epoch - 1] *= 1.0 * epoch / itrcnt
        forward_nfe_arr[epoch - 1] *= 1.0 * epoch / itrcnt
        acc = acc.detach().cpu().numpy() / dsize
        printouts = [epoch, loss_arr[epoch-1], acc, nfe_arr[epoch-1], forward_nfe_arr[epoch - 1], time_arr[epoch-1], (time.time()-start_time)/60]
        print(str_rec(rec_names, printouts, rec_unit, presets="Train|| {}"))
        outlist.append(printouts)
        # writer.writerow(printouts)
        # if epoch % 2 == 0:
        model[1].df.nfe = 0
        test_start_time = time.time()
        loss = 0
        acc = 0
        dsize = 0
        bcnt = 0
        for x, y in tsdat:
            # forward in time and solve ode
            dsize += y.shape[0]
            y = y.to(device=args.gpu)
            pred_y = model(x.to(device=args.gpu)).detach()
            pred_l = torch.argmax(pred_y, dim=1)
            acc += torch.sum((pred_l == y).float())
            bcnt += 1
            # compute loss
            loss += loss_func(pred_y, y).detach() * y.shape[0]
        test_time = time.time() - test_start_time
        loss /= dsize
        acc /= dsize
        printouts = [epoch, loss.detach().cpu().numpy(), acc.detach().cpu().numpy(), str(model[1].df.nfe / bcnt), None, test_time, (time.time()-start_time)/60]
        print(str_rec(rec_names, printouts, presets="Test || {}"))
        outlist.append(printouts)
        # writer.writerow(printouts)
    return outlist
