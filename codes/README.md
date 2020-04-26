# Code Framework

## Consistency Enforcing Module (CEM)
This module can wrap any existing (and even pre-trained) super resolution network, modifying its high-resolution outputs to be consistent with the low-resolution input. More details on the idea behind this module and its derivation can be found in the paper.

#### Initializing the module:
Start by creating a CEM configuration object, which can be initialized to default setting by passing only the super resolution scale factor ``sf``:

```
CEM_config = CEMnet.Get_CEM_Config(sf)
``` 

Then create the CEM object, by passing the assumed upsampling kernel ``upscale_kernel`` as an ND-array. For the default bicubic kernel, exclude this argument:

```
CEM = CEMnet.CEMnet(CEM_config,upscale_kernel=None)
```
#### Wrapping a super resolution model
1. Wrap an existing (pre-trained or not) super-resolution ``SR_model`` with the CEM by typing:

    ```wrapped_model = CEM.WrapArchitecture_PyTorch(SR_model,training_patch_size=None)```

    Argument ``training_patch_size`` is used for creating a mask that discards image regions near the boundaries, that may be invalid due to convolution with the CEM filters. This is usefull when calculating some loss function of the output of the CEM. ``None`` may be passed otherwise.
2. Now run the wrapped model on any low resolution image ``x``, by simply calling ``consistent_SR_im = wrapped_model(x)``. The CEM will run the underlying SR model and modify its output to be consistent with ``x``, given the assumed ``upscale_kernel``.

**Note:**
Calling ``wrapped_model.train(mode)`` invokes the respective command of the underlying ``SR_model``, but also determines whether input ``x`` would be pre-padded (when ``mode==False``) or not (for ``mode==True``), to reduce artifacts resulting from convolution with the CEM filters. Pre-padding and padding removal is done behind the scenes, and does requre any user action.

 
<!--
## Table of Contents
1. [Config](#config)
1. [Data](#data)
1. [Model](#model)
1. [Network](#network)
1. [Utils](#utils)
1. [Scripts](#scripts)

## Config
#### [`options/`](https://github.com/xinntao/BasicSR/tree/master/codes/options) Configure the options for data loader, network structure, model, training strategies and etc.

- `json` file is used to configure options and [`options/options.py`](https://github.com/xinntao/BasicSR/blob/master/codes/options/options.py) will convert the json file to python dict.
- `json` file uses `null` for `None`; and supports `//` comments, i.e., in each line, contents after the `//` will be ignored. 
- Supports `debug` mode, i.e, model name start with `debug_` will trigger the debug mode.
- The configuration file and descriptions can be found in [`options`](https://github.com/xinntao/BasicSR/tree/master/codes/options).

## Data
#### [`data/`](https://github.com/xinntao/BasicSR/tree/master/codes/data) A data loader to provide data for training, validation and testing.

- A separate data loader module. You can modify/create data loader to meet your own needs.
- Uses `cv2` package to do image processing, which provides rich operations.
- Supports reading files from image folder or `lmdb` file. For faster IO during training, recommand to create `lmdb` dataset first. More details including lmdb format, creation and usage can be found in our [lmdb wiki](https://github.com/xinntao/BasicSR/wiki/Faster-IO-speed).
- [`data/util.py`](https://github.com/xinntao/BasicSR/blob/master/codes/data/util.py) provides useful tools. For example, the `MATLAB bicubic` operation; rgbycbcr as MATLAB. We also provide [MATLAB bicubic imresize wiki](https://github.com/xinntao/BasicSR/wiki/MATLAB-bicubic-imresize) and [Color conversion in SR wiki](https://github.com/xinntao/BasicSR/wiki/Color-conversion-in-SR).
- Now, we convert the images to format NCHW, [0,1], RGB, torch float tensor.

## Model
#### [`models/`](https://github.com/xinntao/BasicSR/tree/master/codes/models) Construct different models for training and testing.

- A model mainly consists of two parts - [network structure] and [model defination, e.g., loss definition, optimization and etc]. The network description is in the [Network part](#network).
- Based on the [`base_model.py`](https://github.com/xinntao/BasicSR/blob/master/codes/models/base_model.py), we define different models, e.g., [`SR_model.py`](https://github.com/xinntao/BasicSR/blob/master/codes/models/SR_model.py), [`SRGAN_model.py`](https://github.com/xinntao/BasicSR/blob/master/codes/models/SRGAN_model.py), [`SRRaGAN_model.py`](https://github.com/xinntao/BasicSR/blob/master/codes/models/SRRaGAN_model.py) and [`SFTGAN_ACD_model.py`](https://github.com/xinntao/BasicSR/blob/master/codes/models/SFTGAN_ACD_model.py).

## Network
#### [`models/modules/`](https://github.com/xinntao/BasicSR/tree/master/codes/models/modules) Construct different network architectures.

- The network is constructed in [`models/network.py`](https://github.com/xinntao/BasicSR/blob/master/codes/models/networks.py) and the detailed structures are in [`models/modules`](https://github.com/xinntao/BasicSR/tree/master/codes/models/modules).
- We provide some useful blocks in [`block.py`](https://github.com/xinntao/BasicSR/blob/master/codes/models/modules/block.py) and it is flexible to construct your network structures with these pre-defined blocks.
- You can also easily write your own network architecture in a seperate file like [`sft_arch.py`](https://github.com/xinntao/BasicSR/blob/master/codes/models/modules/sft_arch.py). 
-->
## Utils
[`utils/`](https://github.com/xinntao/BasicSR/tree/master/codes/utils) Provides useful utilities.
<!--
- [logger.py](https://github.com/xinntao/BasicSR/blob/master/codes/utils/logger.py) provides logging service during training and testing.
- Support to use [tensorboard](https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard) to visualize and compare training loss, validation PSNR and etc. Installationand usage can be found [here](https://github.com/xinntao/BasicSR/tree/master/codes/utils).
- [progress_bar.py](https://github.com/xinntao/BasicSR/blob/master/codes/utils/progress_bar.py) provides a progress bar which can print the progress. 
-->
## Scripts
<!--#### [`scripts/`](https://github.com/xinntao/BasicSR/tree/master/codes/scripts) Privide useful scripts.
Details can be found [here](https://github.com/xinntao/BasicSR/tree/master/codes/scripts).
-->
