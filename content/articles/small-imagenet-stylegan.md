---
title: 🖼️ Using ImageNet 32x32 on StyleGAN 2 with ADA and GTX 30xx Series
description: For my current research, I got up and running with StyleGAN 2 ADA and the ImageNet 32x32 dataset. Here's how. 
slug: small-imagenet-stylegan
---

For my current research, I decided to get [StyleGAN 2 ADA](https://arxiv.org/abs/2006.06676) up and running on an Ubuntu 20.04 machine with a GTX 3090 GPU. The kind individuals at NVIDIA made this architecture open-source on GitHub, which made it a simple 
```python
git pull https://github.com/NVlabs/stylegan2-ada-pytorch.git
``` 
to begin working with the GAN. However, I immediately faced some issues when using this code on a 3090 card, and had to do some digging to avoid these issues. Thus, I decided to detail how I got the GAN running on my machine. 

# Getting Started 
First, I created a new [Conda](https://www.anaconda.com/) environment with PyTorch 1.9, and I quickly learned that the library only works with Pytorch 1.7. Thus, I removed `Torch` and its affiliated libraries, and started fresh. Once I had downloaded PyTorch 1.7, I realized that this version was incompatible with the current CUDA install on my machine (version 11.2) as the source code required CUDA of at least 11.1. Not to fear, with a bit of digging through the GitHub issues, I found [this solution](https://github.com/NVlabs/stylegan2-ada-pytorch/issues/97#issuecomment-875325362) which discussed installing PyTorch using PIP using 
```shell 
$ pip install torch===1.7.1+cu110 torchvision===0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```
However, I encountered an error when I attempted to get the model spinning up and training on MNIST for a test run: A message declaring 
```error 
ImportError: No module named 'upfirdn2d_plugin'
```
Fortunately, the solution that I found on [GitHub issues](https://github.com/NVlabs/stylegan2-ada-pytorch/issues/39#issuecomment-903338054) also solved this problem right away: all I had to do was uninstall ninja from Conda, reinstall it, and then run 
```shell 
$ rm -rf ~/.cache/torch_extensions/*
```

## Downloading the Data
Next came the fun part: downloading and training on the data. I started out by downloading the ImageNet 32x32 file from the ImageNet website. You can do so by simply registering [here](https://image-net.org/signup.php) and then going [here](https://image-net.org/download-images) to download the images. Make sure that you download these images into the root directory of the StyleGAN 2: ADA repository that you cloned onto your machine. Once I had the 3 Gb file on my machine, I went ahead and modified the script proposed [here](https://patrykchrabaszcz.github.io/Imagenet32/) to get to work using this data. 

First, we can import the necessary packages. Since the file is stored in a Python Pickle format, we'll have to unpickle it to access the internal data. 

```python
import numpy as np
import pickle
from PIL import Image
import os
import json
```
From here, we can follow the aforementioned tutorial for loading in our data and labels. We'll start by creating a function to unpickle our data. 

```python
# Load in data
def load_pickle(file):
    with open(f'train_data_batch_{file}', 'rb') as f:
        data = pickle.load(f)
        return data
```

Next, we'll go ahead and write a function to fetch the samples as well as their corresponding labels from a specified file.

```python 
def load_data(idx, img_size=32):
    data = load_pickle(idx)
    
    x = data['data']
    y = data['labels']
    
    data_size = x.shape[0]
    y = [i - 1 for i in y]
    
    img_size2 = img_size * img_size 
    
    # Reshape the data
    x = np.dstack((x[:, :img_size2], x[:, img_size2:2*img_size2], x[:, 2*img_size2:]))
    x = x.reshape((x.shape[0], img_size, img_size, 3))
    
    return x_train, y_train
```

Finally, we'll write the script to actually save each image from within the dataset to a folders images, as well as a dataset JSON as specified by the StyleGAN 2: ADA repository. This JSON will store a list of lists that contain an image path as well as its corresponding label.

```python 
def save_data():
    count = 0
    os.mkdir('images')
    
    file_counts = [0 for i in range(1000)]
    labels = {'labels': []}
    
    for i in range(10):
        x, y = load_data(i + 1)
        
        for l in range(len(y)):
            label = y[l]
            img = Image.fromarray(x[l], 'RGB')
            img.save(f'images/{str(count).zfill(9)}.png')
            labels['labels'].append([f'{str(count).zfill(9)}.png', label])
            count += 1       

    with open('images/dataset.json', 'w') as fp:
        json.dump(labels, fp)
```

This script will save all of our data into a folder titled `images`. From here, all we have to do is run this script to unpickle all the ImageNet data and display it. From there, we will go ahead and run the following command to create a dataset compatible with StyleGAN 2: ADA.

```shell
$ python dataset_tool.py --source ./imagenet/images --dest ./imagenet32/
```

Running this command (in the root directory of where StyleGAN 2: ADA is located) will generate a folder at `imagenet32` that contains your dataset in a format that is simple to understand by the training script. From here, it's as simple as running 

```shell
python train.py --outdir=./training-runs --data=./imagenet32 --gpus=1 --cond=1 --cfg=cifar
```

This will begin training a fresh 32x32 StyleGAN with Adaptive Discriminator Augmentations on a single GPU, with the default parameters specified by the NVIDIA team when they initially trained on the CIFAR 10 dataset. Though your images will start out looking like this:

![Yikes](/article4/fakes000000.png)

After about 17 hours of training, it will look like this 

![Better](/article4/fakes010080.png)

# Final Thoughts
To be honest, there's nothing that special about this code. It's more so meant to show you how easy it is to get set up with a state-of-the-art image generation tool to create new data; or, if you're like me, use this to conduct some research on the viability of synthetic data. Don't hesitate to [reach out](mailto:john@sciteens.org?subject=StyleGAN%202%3A%20ADA) to me if you have any questions about the code!


