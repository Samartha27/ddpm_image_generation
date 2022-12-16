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


## Problem statement




## Methodology

In a nutshell, first we try to slowly and systematically corrupt the inherent structure in a data distribution with an iterative forward diffusion process by making use of noise. Next, we try to use a UNET to learn a reverse diffusion process that restores the lost structure in the data distribution. This should yeild us a tractable generative model of the data.

## Dataset
The dataset was downloaded from [Kaggle](https://www.kaggle.com/) 
Training is with the use of 3 channel images have a size of `32x32`.  The images are normalized using mean as ```[0.485, 0.456, 0.406]``` and standard deviation as ```[0.229, 0.224, 0.225]``` 



## Experiments
Our model is trained on NVIDIA GTX 1050 Ti GPU.

## Results



## References

[Diffusion Models](https://medium.com/@monadsblog/diffusion-models-4dbe58489a2f)

[Diffusion Models Made Easy](https://towardsdatascience.com/diffusion-models-made-easy-8414298ce4da)

[EINOPS](https://github.com/arogozhnikov/einops)




