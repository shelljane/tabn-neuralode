# A New Batch Normalization for Neural ODE

## Dependencies

```
torch
torchvision
torchdiffeq
datasets
einops
botorch
gpytorch
torchcubicspline
```

## How to run the experiment for image classification?

```bash
python3 main.py -b 'train batch' -c 'test batch' -d 'dataset' -e 'epochs' -n 'normalization' -m 'model' -r 'weight decay for BN' -l 'load file' -s 'save file'
```

> 'dataset' can be \{'mnist', 'cifar10', 'svhn', 'cifar100', 'tiny-imagenet'\}
> 'normalization' can be \{'batchnorm', 'bilinear', 'gaussian', ...\}, please specify '-r 0.1' for 'bilinear'
> 'model' can be \{'simple0', 'simple1', 'simple2', ..., 'simple8', 'unet', ...\}, it is 'simple1' by default, 'simple0' has a conv2d before neural ode, others are neural ode only

### Example

```bash
python3 main.py -b 256 -c 256 -d cifar10 -e 128 -n bilinear -m simple1 -r 0.1
python3 main.py -b 256 -c 256 -d cifar10 -e 128 -n bilinear -m unet -r 0.1

python3 main.py -b 256 -c 256 -d cifar100 -e 128 -n bilinear -m unet -r 0.1
python3 main.py -b 256 -c 256 -d svhn -e 128 -n bilinear -m unet -r 0.1
python3 main.py -b 256 -c 256 -d tiny-imagenet -e 128 -n bilinear -m unet -r 0.1
python3 main.py -b 256 -c 256 -d mnist -e 128 -n bilinear -m simple2 -r 0.1

python3 main.py -b 256 -c 256 -d cifar10 -e 128 -n batchnorm -m simple1
python3 main.py -b 256 -c 2 -d cifar10 -e 128 -n batchnorm -m simple1

python3 main.py -b 256 -c 256 -d cifar10 -e 128 -n gaussian -m simple1
python3 main.py -b 256 -c 256 -d cifar10 -e 128 -n smooth -m simple1
python3 main.py -b 256 -c 256 -d cifar10 -e 128 -n cubic -m simple1
```

### To run TA-BN integrated into existing Neural ODE variants

```
cd integration
python3 main.py --norm bilinear
python3 main.py --norm nonorm
```

