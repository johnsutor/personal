---
title: "🔀 Replacing pre-trained Pytorch model layers with custom layers"
description: "My contempt for Batch Normalization and love of Dropout layers led me to upgrade pre-trained convolutional neural networks in the simplest fashion possible."

date: "2021-04-11"
---

---

## Introduction

[Pretrained Networks](https://pytorch.org/vision/stable/models.html) in Torchvision are pretty killer, especially when it comes to tasks requiring pre-training (such as computer vision tasks that require synthetic data synthetic data). However, I've always loathed the complexity of swapping out layers, activations, and normalizations within pre-trained networks. Torchvision pretrained models aren't necessarily Sequential models, and thus, it's not as simple as swapping a layer in and out; You can end up ruining the forward pass of a pretrained Pytorch neural network by appending to and augmenting its layers. At the same time, I don't want to recreate an entire network from scratch before training. Rather, I'd prefer to switch out a few layers. For example, I prefer to include Dropout layers after my activations, and I much prefer [Group Normalization](https://openaccess.thecvf.com/content_ECCV_2018/papers/Yuxin_Wu_Group_Normalization_ECCV_2018_paper.pdf) to Batch Normalization. 


## Implementation
Instead of recreating an entire network, copying over each pretrained layer and defining a new forward method, I'd much prefer to change a network in place. Doing so is pretty simple, and only requires a bit of module tweaking. One of the easiest layer types to replace are the activations of a Pytorch convolutional neural network. Their initializations don't require information regarding the number of channels passed from the previous layer or the number of channels that layer is outputting (which is the case for some normalization layers and convolutional layers). Instead, we can simply pass an instance to our new network class, as well as the pretrained network and the target layers that we wish to replace. In my case, I prefer to insert Dropout layers after activation layers. Thus, I create a *FusedReLU* class to swap into pre-trained networks.

```python
import torch.nn as nn

class FusedReLU(nn.Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p)
        
    def forward(self, x):
        x = self.relu(x)
        x = self.dropout(x)
        return x
```

From this, we can create our modified pre-trained network by swapping out every Rectified Linear Unit activation with our FusedReLU class. 

```python
from functools import reduce

class ReplacementNet(nn.Module):
    
    def rgetattr(self, obj, attr, *args):
        _getattr = lambda obj, attr, *args: getattr(obj, attr, *args)
        return reduce(_getattr, [obj] + attr.split('.'))
    
    def rsetattr(self, obj, attr, val):
        pre, _, post = attr.rpartition('.')
        return setattr(self.rgetattr(obj, pre) if pre else obj, post, val)
    
    def __init__(self, model: nn.Module, target: nn.Module, replacement: nn.Module):
        """
        Args:
            model: (nn.Module) The pre-initialized neural network to find and replace each target layer.
            target: (nn.Module) The target layer to replace each instance of. 
            replacement: (nn.Module) Thelayer (or sequential model) to replace the target layer with. 
            
        """
        super().__init__()
        self.model = model
        
        for child in dict(self.model.named_modules()).items():
            if child[0]:
                module = self.rgetattr(model, child[0])
                if module._get_name() == target()._get_name():
                    self.rsetattr(model, child[0], replacement())
                    
    def forward(self, x):
        x = self.model(x)
        return x
    
model = ReplacementNet(resnet18(pretrained=True), nn.ReLU, FusedReLU)
```

Here, we iterate over each module and nested module using the `reduce()` function provided by the built-in functools library. As discussed [here](https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-subobjects-chained-properties), we can use this method to set nested attributes and get nested attributes for our network. If we inspect our ResNet18 base initialization, as well as our ReplacementNet version of the ResNet18, we can see that each ReLU layer is replaced by a FusedReLU (I'll spare copying and pasting the output here so that your entire screen isn't clogged with quasi-useless information). Et viola! We now have a pre-trained ResNet18 with dropout enabled after each activation.

Introducing new normalization layers to our network requires a marginal amount of tweaking to the snippet above. All we have to do is check the number of channels taken in by the existing normalization layers. Thus, our class becomes: 

```python
from functools import reduce

class ReplacementNet(nn.Module):
    
    def rgetattr(self, obj, attr, *args):
        _getattr = lambda obj, attr, *args: getattr(obj, attr, *args)
        return reduce(_getattr, [obj] + attr.split('.'))
    
    def rsetattr(self, obj, attr, val):
        pre, _, post = attr.rpartition('.')
        return setattr(self.rgetattr(obj, pre) if pre else obj, post, val)
    
    def __init__(self, model: nn.Module, target: nn.Module, replacement: nn.Module):
        """
        Args:
            model: (nn.Module) The pre-initialized neural network to find and replace each target layer.
            target: (nn.Module) The target layer to replace each instance of. 
            replacement: (nn.Module) Thelayer (or sequential model) to replace the target layer with. 
            
        """
        super().__init__()
        self.model = model
        
        for child in dict(self.model.named_modules()).items():
            if child[0]:
                module = self.rgetattr(model, child[0])
                if module._get_name() == target(1)._get_name():
                    num_features = module.num_features
                    self.rsetattr(model, child[0], replacement(8, num_features))
                    
    def forward(self, x):
        x = self.model(x)
        return x
    
model = ReplacementNet(resnet18(pretrained=True), nn.BatchNorm2d, nn.GroupNorm)
```

In this case, I simply fetch the number of features that each Batch Normalization layer takes in before initializing a Group Normalization layer with eight separate groups. Note that I initialize a base target that expects a channel size of one so that I can access the `_get_name()` method.

Finally, I can replace certain convolutional layers with a custom convolutional layer. For instance, if we wanted to replace the ResNet18's convolutions with kernel size one with convolutions of kernel size three and padding size one (to preserve the shape of the input), we could do so with the following code. 

```python
from functools import reduce

class ReplacementNet(nn.Module):
    
    def rgetattr(self, obj, attr, *args):
        _getattr = lambda obj, attr, *args: getattr(obj, attr, *args)
        return reduce(_getattr, [obj] + attr.split('.'))
    
    def rsetattr(self, obj, attr, val):
        pre, _, post = attr.rpartition('.')
        return setattr(self.rgetattr(obj, pre) if pre else obj, post, val)
    
    def __init__(self, model: nn.Module, target: nn.Module, replacement: nn.Module):
        """
        Args:
            model: (nn.Module) The pre-initialized neural network to find and replace each target layer.
            target: (nn.Module) The target layer to replace each instance of. 
            replacement: (nn.Module) Thelayer (or sequential model) to replace the target layer with. 
            
        """
        super().__init__()
        self.model = model
        
        for child in dict(self.model.named_modules()).items():
            if child[0]:
                module = self.rgetattr(model, child[0])
                if module._get_name() == target(1,1,1)._get_name() and module.kernel_size == (1,1):
                    in_channels = module.in_channels
                    out_channels = module.out_channels
                    self.rsetattr(model, child[0], replacement(in_channels, out_channels, 3, padding=1, stride=2))
                    
    def forward(self, x):
        x = self.model(x)
        return x
    
model = ReplacementNet(resnet18(pretrained=True), nn.Conv2d, nn.Conv2d)
```

Once again, I had to provide code to create toy initializations for the line `target._get_name()`. To avoid this, it is possible to create a class capable of taking care of built-in pytorch layers passed to the function. In our case, we'll create a class capable of handling multiple types of Pytorch layers. 

However, there are quite a few different types of layers to handle, each contained in the modules:
- activation
- adaptive
- batchnorm
- channelshuffle
- container
- conv
- distance
- dropout
- flatten
- fold
- instancenorm
- linear
- loss
- module
- normalization
- padding
- pixelshuffle
- pooling
- rnn
- sparse
- transformer
- upsampling
- utils

So for this post's sake, we'll take care of 2D convolutions, 2D Batch normalization, and activations. 

```python
from functools import reduce

class ReplacementNet(nn.Module):
    
    def rgetattr(self, obj, attr, *args):
        _getattr = lambda obj, attr, *args: getattr(obj, attr, *args)
        return reduce(_getattr, [obj] + attr.split('.'))
    
    def rsetattr(self, obj, attr, val):
        pre, _, post = attr.rpartition('.')
        return setattr(self.rgetattr(obj, pre) if pre else obj, post, val)
    
    def __init__(self, model: nn.Module, target: nn.Module, replacement: nn.Module, **kwargs):
        """
        Args:
            model: (nn.Module) The pre-initialized neural network to find and replace each target layer.
            target: (nn.Module) The target layer to replace each instance of. 
            replacement: (nn.Module) Thelayer (or sequential model) to replace the target layer with. 
        Keyword Args:
            kernel_size: The kernel size of the replacement convolution.
            target_kernel_size: The kernel size of the target convolution.
                        
        """
        super().__init__()
        self.model = model
        
        for child in dict(self.model.named_modules()).items():
            if child[0]:
                module = self.rgetattr(model, child[0])
                if module._get_name() == target.__name__:
                    if module.__module__.endswith("conv"):
                        if "target_kernel_size" in kwargs and module.kernel_size == kwargs["target_kernel_size"]:
                            in_channels = module.in_channels
                            out_channels = module.out_channels 
                            self.rsetattr(model, child[0], replacement(in_channels, out_channels, kwargs["kernel_size"]))
                    elif module.__module__.endswith("batchnorm"):
                        num_features = module.num_features 
                        self.rsetattr(model, child[0], replacement(num_features))
                    elif module.__module__.endswith("activation"):
                        self.rsetattr(model, child[0], replacement())
                    else:
                        raise NotImplementedError("This layer hasn't been implemented yet")
                    
    def forward(self, x):
        x = self.model(x)
        return x
    
model = ReplacementNet(resnet18(pretrained=True), nn.Conv2d, nn.Conv2d, kernel_size=3)
```

## Final Remarks

That's all for the sake of this post. Hypothetically, you could implement handlers for each module type as shown above. Furthermore, you could extend the list of keyword arguments to allow for a high degree of customization of layers similarly to as I've done for *target_kernel_size* and *kernel_size*. For most cases, though, replacing only a few normalization and activation layers is sufficient.

If you have any recommendations or better ways for achieving what I've outlined above, don't hesitate to [reach out](mailto:john@sciteens.org?subject=Replacing%20Pytorch%20Model%20Layers%20Blog)! 