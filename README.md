# Camera-Independent Monocular Depth Prediction using Metric Space Inputs

By [Jeshwanth Pilla](https://www.linkedin.com/in/jeshwanth-p/), [Helisa Dhamo](http://campar.in.tum.de/Main/HelisaDhamo)

## Contents
0. [Introduction](#introduction)
0. [Quick Guide](#quick-guide)
0. [Results](#results)
0. [License](#license)


## Introduction

This work is an extension to the existing state-of-art methods for Depth Prediction from a single RGB image. These methods suffers from the problem that a network trained on images from one camera does not generalize well to images taken with a different camera model. Hence, in this work, we tried to propose a new approach that applies metric space transformation to image pixels before training and thereby considering of camera parameters of the dataset. So we used CNN model of Laina et al for this implementation, as described in the paper "[Deeper Depth Prediction with Fully Convolutional Residual Networks](https://arxiv.org/abs/1606.00373)". Then we try to compare the evaluation results of this proposed approach over the one without this transformation on benchmark dataset NYU Depth v2 for indoor scenes. Our experiments show that neural networks tend to perform better with camera coordinates appended compared to one without, indicating the improvement in networks generalizing to images of different camera models. 

## Quick Guide

The trained model is in TensorFlow Framework v1. Please read below for more information on how to get started.

### Prerequisites

[NYU Labelled Dataset](http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat) <br />
Official NYU train-test split (Included in NYU_Data Folder)<br />
Kinect Camera Parameters (Included in NYU_Data Folder)<br />
Pretrained ResNet weight checkpoint

### TensorFlow
The code provided in the *tensorflow* folder requires accordingly a successful installation of the [TensorFlow](https://www.tensorflow.org/) library (any platform). 
The model's graph is constructed in ```fcrn.py``` and we used pretrained ResNet weights for training.<br />
The code for training is found in ```train.py``` and and hyperparameters can be changed there. Use ```python train.py resnet-weights.ckpt NYUImageFolderPath OfficialSplitFilePath``` to try the code.<br />
```predict.py``` provides sample code for using the network to predict the depth map of an input image. Use ```python predict.py NYU_FCRN_model.ckpt yourimage.jpg``` to try the code. <br />
For acquiring the evaluation on NYU or Sun3D, the user can run  `evaluate.py` or `evaluateSun3D.py` respectively.


## Results

In the following table, we report the results of proposed metric model compared with that of baseline model that we obtained after evaluation
- Error metrics on NYU Depth v2:

| State of the art on NYU     |  rel  |  rms  | log10 |
|-----------------------------|:-----:|:-----:|:-----:|
| Baseline model		| 0.3728 | 1.3699 | 0.1769 |  
| **Proposed Metric Model**	| **0.3671** | **1.3523** | **0.1720** |

- Qualitative results:

![Results](https://github.com/JeshDev/FCRN-DepthPrediction/blob/master/results.png)

## License

Simplified BSD License

Copyright (c) 2016, Iro Laina  
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
