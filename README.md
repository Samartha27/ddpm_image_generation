# Denoising Diffusion Probabilistic Model with Pytorch

<!---
## Installation

```bash
$ git clone https://github.com/Samartha27/mini_ddmp.git
$ cd mini_ddmp
$ conda env create -f environment.yml
$ conda activate env
```
--->


## Abstract

In the Denoising Diffusion Probabilistic model for image generation, we try to slowly and systematically corrupt the inherent structure in a data distribution with an iterative forward diffusion process by making use of noise. Next, we try to use a UNET to learn a reverse diffusion process that restores the lost structure in the data distribution. This should yeild us a tractable generative model of the data.


## Problem statement




## Methodology

We start with the original image and iteratively add noise in each step. After sufficient number of steps we say that the image is nothing but pure noise.  We use a Normal distribution to sample the noise. We do not employ the same noise at each timestep during the forward process. This can be regulated with the help of the Linear Scheduler which scales the mean and variance inorder to avoid variance explosion as the noise is increases.  The reverese diffusion process involves the neural network trying to learn how to remvove noies step by step.  This way after the model has completed trying, when we feed the model pure noise sampled from the Normal Distribution, it gradually removes the noise in specified timesteps for tractable outcome and prodcues the output image with clarity. 
The model produces 3 things:
1. Mean of the noise at each time step. The variance is kept fized in this implementation
2. Predicting the original image directly (Not practical)
3. The noise of image directly

The U-net architecture takes the input image and projects the image into samller resolution bottleneck with the help of a Resnet block and Downsample block. After the bottleneck it projects the module back into the original size with the help of Upsample blocks. There are attention blocks employed at certain resolutions along with skip connections etween layers of the same spatial resolutions.The sinusoidal embeddings projected into each of the residual blocks informs the model of which timestep it is running and also helps the model during the reverese-diffusion / denoising process to remove appropriate amounts of noise corresponding to how much noise was added in the forward diffusion at each time step.


## Dataset

For training we used the [TinyImageNet](https://courses.cs.washington.edu/courses/cse599g1/19au/files/homework2.tar) dataset with size `64 x 64 x 3`.  The images are normalized using mean as ```[0.485, 0.456, 0.406]``` and standard deviation as ```[0.229, 0.224, 0.225]``` 



## Experiments
Our model is trained on NVIDIA GTX 1050 Ti GPU.




## Results





## References

[Diffusion Models](https://medium.com/@monadsblog/diffusion-models-4dbe58489a2f) <br />
[Diffusion Models Made Easy](https://towardsdatascience.com/diffusion-models-made-easy-8414298ce4da) <br />
[EINOPS](https://github.com/arogozhnikov/einops)




