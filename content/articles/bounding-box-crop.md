---
title: 🖼️ Bounding box cropping in Python
description: This simple algorithm creates square crops about a bounding box for use in Deep Learning pipelines. 
slug: bounding-box-crop
---

Recently, while working on a deep learning research project leveraging GANs, I ran into the issue of non-uniform image sizes. 
I was working with the [DEIC CUB Dataset](https://github.com/cvjena/deic/tree/master/datasets/cub), which is a variant of the CalTech-UCSD Birds dataset that strictly limits the number of training images per class. One of the larger issues when working with the CUB dataset is that the images are of non-uniform dimensions, and many of the objects (well, birds) aren't dead center of each photo. As is the case for the majority of GAN architectures, the input is required to be a square image that has a resolution that is a power of two. Unfortunately, for the DEIC CUB dataset, only 2.2% of the images are a square ratio, with the majority of the images either being of the ratio of 500:333 (18.0%) or 4:3 (11.6%). Thus, it's necessary to transform the dataset into images that are square, where each side is a power of two length, and where the foreground object (again, a bird) is within the frame of the crop. I couldn't really find any algorithms out there that were sufficient for this task, so I mocked one up for the purposes of the project. 

# The algorithm

In the algorithm, we require the center coordinates of the bounding box (x and y), the width and the height of the bounding box (w and h), as well as the final shape of the image. I opted out of using an assertion to check that the value provided is a multiple of two because the algorithm is general purpose. The first step is to determine whether or not to scale the image. To do so, the algorithm finds the minimum of the ratios of the height and width of the original image to the proposed shape of the image. Afterward, the image is scaled accordingly. If needed, you could change the interpolation code here to another method; the list of downsampling interpolation methods are listed [here](https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-filters). 

From here, the algorithm scales the image and recalculates the bounding box center coordinates according to the new scale. After this step, the algorithm calculates the pixel edges of the crop that we wish to create for our image. If the pixel edges are less than or greater than the edges of the image, we can recalculate them to still ensure that the object is in frame. Furthermore, if the cropped image isn't a square ratio in the end, we can simply pad the sides of the image so that the final image is a square. We do so using the [NumPy Pad](https://numpy.org/doc/stable/reference/generated/numpy.pad.html) function. I opted for a reflection pad for the image, though you can change it to any of the options available to the NumPy padding function as listed above. Finally, we crop and return the image. I've provided some samples and their cropped counterparts, as well as the code, below. If you have any feedback for the algorithm, don't hesitate to [contact me](mailto:john@sciteens.org)!

### Originals
![Albatross](/article6/albatross_before.jpg)
![Raven](/article6/raven_before.jpg)
![Warbler](/article6/warbler_before.jpg)

### Cropped (shape = 256)
![Albatross](/article6/albatross_after.jpg)
![Raven](/article6/raven_after.jpg)
![Warbler](/article6/warbler_after.jpg)




### Code
<Gist id="johnsutor/9d1c332ef0a042e01a70f2b511b85f65" />


