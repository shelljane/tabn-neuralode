import torch
from torch.utils import data
import torchvision as tv
import datasets
import numpy as np

def loaders(name="cifar10", batchTrain=256, batchTest=2, njobs=8, noise=0, nratio=0.1): 
    if name == "mnist": 
        preprocTrain = tv.transforms.Compose([
                tv.transforms.ToTensor(), 
                tv.transforms.Normalize(mean=[0.5, ],std=[0.5, ]),
                tv.transforms.RandomHorizontalFlip(), 
                tv.transforms.RandomErasing(scale=(0.04, 0.2), ratio=(0.5, 2)), 
                tv.transforms.RandomCrop(28, padding=4)])
        preprocTest = tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=[0.5, ],std=[0.5, ])])
        trainset = tv.datasets.MNIST(root="./data", train=True, download=True, transform=preprocTrain)
        testset = tv.datasets.MNIST(root="./data", train=False, download=True, transform=preprocTest)
        loaderTrain = data.DataLoader(trainset, batch_size=batchTrain, shuffle=True, num_workers=njobs)
        loaderTest = data.DataLoader(testset, batch_size=batchTest, shuffle=False, num_workers=njobs)
    elif name == "cifar10": 
        preprocTrain = tv.transforms.Compose([
                tv.transforms.ToTensor(), 
                tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
                tv.transforms.RandomHorizontalFlip(), 
                tv.transforms.RandomErasing(scale=(0.04, 0.2), ratio=(0.5, 2)), 
                tv.transforms.RandomCrop(32, padding=4)])
        preprocTest = tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        trainset = tv.datasets.CIFAR10(root="./data", train=True, download=True, transform=preprocTrain)
        testset = tv.datasets.CIFAR10(root="./data", train=False, download=True, transform=preprocTest)
        if noise > 0: 
            trainset.data = trainset.data.astype(np.float64)
            testset.data = testset.data.astype(np.float64)
            for idx in range(trainset.data.shape[0]): 
                shape = trainset.data[idx].shape
                if np.random.rand() < nratio: 
                    trainset.data[idx] += noise * np.random.randn(*shape)
            for idx in range(testset.data.shape[0]): 
                shape = testset.data[idx].shape
                if np.random.rand() < nratio: 
                    testset.data[idx] += noise * np.random.randn(*shape)
            trainset.data = trainset.data.astype(np.uint8)
            testset.data = testset.data.astype(np.uint8)
        loaderTrain = data.DataLoader(trainset, batch_size=batchTrain, shuffle=True, num_workers=njobs)
        loaderTest = data.DataLoader(testset, batch_size=batchTest, shuffle=False, num_workers=njobs)
    elif name == "cifar100": 
        preprocTrain = tv.transforms.Compose([
                tv.transforms.ToTensor(), 
                tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
                tv.transforms.RandomHorizontalFlip(), 
                tv.transforms.RandomErasing(scale=(0.04, 0.2), ratio=(0.5, 2)), 
                tv.transforms.RandomCrop(32, padding=4)])
        preprocTest = tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        trainset = tv.datasets.CIFAR100(root="./data", train=True, download=True, transform=preprocTrain)
        testset = tv.datasets.CIFAR100(root="./data", train=False, download=True, transform=preprocTest)
        loaderTrain = data.DataLoader(trainset, batch_size=batchTrain, shuffle=True, num_workers=njobs)
        loaderTest = data.DataLoader(testset, batch_size=batchTest, shuffle=False, num_workers=njobs)
    elif name == "svhn": 
        preprocTrain = tv.transforms.Compose([
                tv.transforms.ToTensor(), 
                tv.transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5]),
                tv.transforms.RandomHorizontalFlip(), 
                tv.transforms.RandomErasing(scale=(0.04, 0.2), ratio=(0.5, 2)), 
                tv.transforms.RandomCrop(32, padding=4)])
        preprocTest = tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        trainset = tv.datasets.SVHN(root="./data", split="train", download=True, transform=preprocTrain)
        testset = tv.datasets.SVHN(root="./data", split="test", download=True, transform=preprocTest)
        loaderTrain = data.DataLoader(trainset, batch_size=batchTrain, shuffle=True, num_workers=njobs)
        loaderTest = data.DataLoader(testset, batch_size=batchTest, shuffle=False, num_workers=njobs)
    elif name == "tiny-imagenet": 
        preprocTrain = tv.transforms.Compose([
                tv.transforms.ToTensor(), 
                tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
                tv.transforms.RandomHorizontalFlip(), 
                tv.transforms.RandomErasing(scale=(0.04, 0.2), ratio=(0.5, 2)), 
                tv.transforms.RandomCrop(64, padding=8)])
        preprocTest = tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        def transformsTrain(examples):
            examples["pixel_values"] = [preprocTrain(image.convert("RGB")) for image in examples["image"]]
            return examples
        def transformsTest(examples):
            examples["pixel_values"] = [preprocTest(image.convert("RGB")) for image in examples["image"]]
            return examples
        trainset = datasets.load_dataset("Maysee/tiny-imagenet", split="train")
        testset = datasets.load_dataset("Maysee/tiny-imagenet", split="valid")
        trainset.set_transform(transformsTrain)
        testset.set_transform(transformsTest)
        loaderTrain = data.DataLoader(trainset, batch_size=batchTrain, shuffle=True, num_workers=njobs, 
                                    collate_fn=lambda x: (torch.stack(list(map(lambda y: y['pixel_values'], x)), dim=0), torch.tensor(list(map(lambda y: y['label'], x)), dtype=torch.int64)))
        loaderTest = data.DataLoader(testset, batch_size=batchTest, shuffle=False, num_workers=njobs, 
                                    collate_fn=lambda x: (torch.stack(list(map(lambda y: y['pixel_values'], x)), dim=0), torch.tensor(list(map(lambda y: y['label'], x)), dtype=torch.int64)))

    return loaderTrain, loaderTest



def channel(name="cifar10"): 
    result = None
    if name in ["mnist", ]: 
        result = 1
    elif name in ["cifar10", "cifar100", "svhn"]: 
        result = 3
    elif name in ["tiny-imagenet", ]: 
        result = 3
    return result



def size(name="cifar10"): 
    result = None
    if name in ["mnist", ]: 
        result = 1*28*28
    elif name in ["cifar10", "cifar100", "svhn"]: 
        result = 3*32*32
    elif name in ["tiny-imagenet", ]: 
        result = 3*64*64
    return result


def classes(name="cifar10"): 
    result = None
    if name in ["mnist", "cifar10", "svhn"]: 
        result = 10
    elif name in ["cifar100", ]: 
        result = 100
    elif name in ["tiny-imagenet", ]: 
        result = 200
    return result


if __name__ == "__main__": 
    print(f"Loading dataset")
    loaderTrain, loaderTest = loaders("tiny-imagenet")
    for images, labels in loaderTrain: 
        print(images.shape)
        print(labels.shape)
        exit()
