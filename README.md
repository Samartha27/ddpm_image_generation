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

In the Denoising Diffusion Probabilistic model for image generation, we try to slowly and systematically corrupt the inherent structure in a data distribution with an iterative forward diffusion process by making use of noise. Next, we try to use a U-net to learn a reverse diffusion process that restores the lost structure in the data distribution. This should yeild us a tractable generative model of the data.


## Problem statement




## Methodology

We start with the original image and iteratively add noise in each step. After sufficient iterations we say that the final image follows an isotropoic gaussian.  We use a Normal distribution to sample the noise. We do not employ the same noise at each timestep during the forward process. This can be regulated with the help of the Linear Scheduler which scales the mean and variance inorder to avoid variance explosion as the noise is increases.  The reverse diffusion process involves the neural network trying to learn how to remove noise step by step.  This way after the model has completed trying, when we feed the model pure noise sampled from the Normal Distribution, it gradually removes the noise in specified timesteps for tractable outcome and produces the output image with clarity. 
The model produces 3 predictions:

1. Mean of the noise at each time step. (Variance is kept fixed in this implementation)
2. Predicting the original image directly (Not practical)
3. The noise of image directly

The U-net architecture takes the input image and projects the image into samller resolution bottleneck with the help of a Resnet block and Downsample block. After the bottleneck it projects the module back into the original size with the help of Upsample blocks. There are attention blocks employed at certain resolutions along with skip connections between layers of the same spatial resolutions.The sinusoidal embeddings projected into each of the residual blocks informs the model of which timestep it is running and also helps the model during the Reverse-diffusion / Denoising process to remove appropriate amounts of noise corresponding to how much noise was added in the forward diffusion at each time step.

## Forward Process

We can sample $x_t$ at any timestep $t$ with,

$$
\begin{align}
q(x_t|x_0) &= \mathcal{N} \Big(x_t; \sqrt{\bar\alpha_t} x_0, (1-\bar\alpha_t) \mathbf{I} \Big)
\end{align}$$

where $\alpha_t = 1 - \beta_t$ and $\bar\alpha_t = \prod_{s=1} ^{t} \alpha_s$

## Reverse Process
The reverse process removes noise starting at $p(x_T) = \mathcal{N}(x_T; \mathbf{0}, \mathbf{I})$
for $T$ time steps.

$$
\begin{align}
p_\theta(x_{t-1} | x_t) &= \mathcal{N}\big(x_{t-1};
\mu_\theta(x_t, t), \Sigma_\theta(x_t, t)\big) \\
p_\theta(x_{0:T}) &=p_\theta(x_T) \prod_{t = 1}^{T} p_\theta(x_{t-1} | x_t) \\
\end{align}$$

where, $\theta$ are the learnable parameters.



Predicting noise:

$$
\begin{align}
\mu_\theta(x_t, t) &= \tilde\mu \bigg(x_t,
  \frac{1}{\sqrt{\bar\alpha_t}} \Big(x_t -
   \sqrt{1-\bar\alpha_t} \epsilon_\theta(x_t, t) \Big) \bigg) \\
  &= \frac{1}{\sqrt{\alpha_t}} \Big(x_t -
  \frac{\beta_t}{\sqrt{1-\bar\alpha_t}} \epsilon_\theta(x_t, t) \Big)
\end{align}$$

where $\epsilon_\theta$ is a learned function that predicts $\epsilon$ given $(x_t, t)$.



















## Dataset

For training we used the [TinyImageNet](https://courses.cs.washington.edu/courses/cse599g1/19au/files/homework2.tar) dataset with size `64 x 64 x 3`.  The images are normalized using mean as ```[0.485, 0.456, 0.406]``` and standard deviation as ```[0.229, 0.224, 0.225]``` 



## Experiments
Our model is trained on NVIDIA GTX 1050 Ti GPU.




## Results





## References

[Diffusion Models](https://medium.com/@monadsblog/diffusion-models-4dbe58489a2f) <br />
[Diffusion Models Made Easy](https://towardsdatascience.com/diffusion-models-made-easy-8414298ce4da) <br />
[EINOPS](https://github.com/arogozhnikov/einops)




