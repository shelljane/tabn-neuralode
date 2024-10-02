import sys
sys.path.append(".")
import time
import pickle
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from tqdm import tqdm

import net
import bench

def parseArgs(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--accum", type=int, default=1)
    parser.add_argument("-b", "--batchTrain", type=int, default=256)
    parser.add_argument("-c", "--batchTest", type=int, default=256)
    parser.add_argument("-d", "--dataset", type=str, default="cifar10")
    parser.add_argument("-e", "--epochs", type=int, default=128)
    parser.add_argument("-n", "--norm", type=str, default="gaussian")
    parser.add_argument("-m", "--model", type=str, default="simple")
    parser.add_argument("-r", "--bndecay", type=float, default=0)
    parser.add_argument("-s", "--save", type=str, default="data/saved.pth")
    parser.add_argument("-l", "--load", type=str, default="")
    parser.add_argument("-j", "--njobs", type=int, default=8)
    parser.add_argument("--noise", type=float, default=0)

    return parser.parse_args()


if __name__ == "__main__": 
    args = parseArgs()

    loaderTrain, loaderTest = bench.loaders(name=args.dataset, batchTrain=args.batchTrain, batchTest=args.batchTest, njobs=args.njobs, noise=args.noise)
    
    channel = bench.channel(args.dataset)
    size = bench.size(args.dataset)
    classes = bench.classes(args.dataset)
    model = net.nodenet(name=args.model, channel=channel, size=size, classes=classes, ntype=args.norm).to(DEVICE)

    opt = optim.AdamW(model.parameters(), lr=1e-3/args.accum, weight_decay=2e-2)
    sched = optim.lr_scheduler.StepLR(opt, round(args.epochs/2), 0.1)
    
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
    
    if len(args.load) > 0: 
        model.load_state_dict(torch.load(args.load))
    
    lossesTrain = []
    accusTrain = []
    lossesTest = []
    accusTest = []
    for epoch in range(args.epochs): 
        print(f"Epoch {epoch}")
        
        model.train()
        
        timeBegin = time.time()
        losses = []
        accus = []
        step = 0
        opt.zero_grad()
        progress = tqdm(loaderTrain)
        for image, label in progress: 
            image, label = image.to(DEVICE), label.to(DEVICE)
            pred = model(image)
            loss = F.cross_entropy(pred, label) + args.bndecay * bnloss()

            loss.backward()
            if step % args.accum == args.accum - 1: 
                nn.utils.clip_grad_norm_(model.parameters(), 10.0*args.accum)
                nn.utils.clip_grad_norm_(bnweights, 1.0*args.accum) if len(bnweights) > 0 else None
                nn.utils.clip_grad_norm_(bnbiases, 1.0*args.accum) if len(bnbiases) > 0 else None
                opt.step()
                opt.zero_grad()
            step += 1

            pred = pred.argmax(dim=-1)
            accu = (pred == label).to(torch.float32).mean()
            losses.append(loss.item())
            accus.append(accu.item())
            lossesTrain.append(loss.item())
            accusTrain.append(accu.item())
            progress.set_postfix(loss=loss.item(), acc=accu.item())
        timeEnd = time.time()
        print(f" -> Training loss={np.mean(losses):.3f}, acc={np.mean(accus):.3f}, time={timeEnd-timeBegin:.3e}s. ") 

        torch.save(model.state_dict(), args.save)

        model.eval()
        with torch.no_grad(): 
            timeBegin = time.time()
            losses = []
            accus = []
            progress = tqdm(loaderTest)
            for image, label in progress: 
                image, label = image.to(DEVICE), label.to(DEVICE)
                pred = model(image)
                loss = F.cross_entropy(pred, label)

                pred = pred.argmax(dim=-1)
                accu = (pred == label).to(torch.float32).mean()
                losses.append(loss.item())
                accus.append(accu.item())
                lossesTest.append(loss.item())
                accusTest.append(accu.item())
                progress.set_postfix(loss=loss.item(), acc=accu.item())
            timeEnd = time.time()
            print(f" -> Testing loss={np.mean(losses):.3f}, acc={np.mean(accus):.3f}, time={timeEnd-timeBegin:.3e}s. ") 

        if sched.get_last_lr()[0] > 1e-4: 
            sched.step()
        net.setEpoch(epoch)
    
    