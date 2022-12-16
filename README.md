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

1. Mean of the noise at each time step. (variance is kept fixed in this implementation)
2. Predicting the original image directly (Not practical)
3. The noise of image directly

The U-net architecture takes the input image and projects the image into samller resolution bottleneck with the help of a Resnet block and Downsample block. After the bottleneck it projects the module back into the original size with the help of Upsample blocks. There are attention blocks employed at certain resolutions along with skip connections between layers of the same spatial resolutions.The sinusoidal embeddings projected into each of the residual blocks informs the model of which timestep it is running and also helps the model during the Reverse-diffusion / Denoising process to remove appropriate amounts of noise corresponding to how much noise was added in the forward diffusion at each time step.

## Forward Process
The forward process adds noise to the data $x_0 \sim q(x_0)$, for $T$ timesteps.

$$
\begin{align}
q(x_t | x_{t-1}) = \mathcal{N}\big(x_t; \sqrt{1-  \beta_t} x_{t-1}, \beta_t \mathbf{I}\big) \\
q(x_{1:T} | x_0) = \prod_{t = 1}^{T} q(x_t | x_{t-1})
\end{align}
$$

where $\beta_1, \dots, \beta_T$ is the variance schedule.

We can sample $x_t$ at any timestep $t$ with,

$$
\begin{align}
q(x_t|x_0) &= \mathcal{N} \Big(x_t; \sqrt{\bar\alpha_t} x_0, (1-\bar\alpha_t) \mathbf{I} \Big)
\end{align}$$

where $\alpha_t = 1 - \beta_t$ and $\bar\alpha_t = \prod_{s=1}^t \alpha_s$

## Reverse Process
The reverse process removes noise starting at $p(x_T) = \mathcal{N}(x_T; \mathbf{0}, \mathbf{I})$
for $T$ time steps.

$$
\begin{align}
p_\theta(x_{t-1} | x_t) &= \mathcal{N}\big(x_{t-1};
\mu_\theta(x_t, t), \Sigma_\theta(x_t, t)\big) \\
p_\theta(x_{0:T}) &=p_\theta(x_T) \prod_{t = 1}^{T} p_\theta(x_{t-1} | x_t) \\
p_\theta(x_0) &= \int {p_\theta}(x_{0:T}) dx_{1:T}
\end{align}$$

$\theta$ are the parameters we train.
## Loss
We optimize the ELBO (from Jenson's inequality) on the negative log likelihood.
\begin{align}
\mathbb{E}[-\log \textcolor{lightgreen}{p_\theta}(x_0)]
 &\le \mathbb{E}_q [ -\log \frac{\textcolor{lightgreen}{p_\theta}(x_{0:T})}{q(x_{1:T}|x_0)} ] \\
 &=L
\end{align}
The loss can be rewritten as  follows.
\begin{align}
L
 &= \mathbb{E}_q [ -\log \frac{\textcolor{lightgreen}{p_\theta}(x_{0:T})}{q(x_{1:T}|x_0)} ] \\
 &= \mathbb{E}_q [ -\log p(x_T) - \sum_{t=1}^T \log \frac{\textcolor{lightgreen}{p_\theta}(x_{t-1}|x_t)}{q(x_t|x_{t-1})} ] \\
 &= \mathbb{E}_q [
  -\log \frac{p(x_T)}{q(x_T|x_0)}
  -\sum_{t=2}^T \log \frac{\textcolor{lightgreen}{p_\theta}(x_{t-1}|x_t)}{q(x_{t-1}|x_t,x_0)}
  -\log \textcolor{lightgreen}{p_\theta}(x_0|x_1)] \\
 &= \mathbb{E}_q [
   D_{KL}(q(x_T|x_0) \Vert p(x_T))
  +\sum_{t=2}^T D_{KL}(q(x_{t-1}|x_t,x_0) \Vert \textcolor{lightgreen}{p_\theta}(x_{t-1}|x_t))
  -\log \textcolor{lightgreen}{p_\theta}(x_0|x_1)]
\end{align}
$D_{KL}(q(x_T|x_0) \Vert p(x_T))$ is constant since we keep $\beta_1, \dots, \beta_T$ constant.
### Computing $L_{t-1} = D_{KL}(q(x_{t-1}|x_t,x_0) \Vert \textcolor{lightgreen}{p_\theta}(x_{t-1}|x_t))$
The forward process posterior conditioned by $x_0$ is,
\begin{align}
q(x_{t-1}|x_t, x_0) &= \mathcal{N} \Big(x_{t-1}; \tilde\mu_t(x_t, x_0), \tilde\beta_t \mathbf{I} \Big) \\
\tilde\mu_t(x_t, x_0) &= \frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1 - \bar\alpha_t}x_0
                         + \frac{\sqrt{\alpha_t}(1 - \bar\alpha_{t-1})}{1-\bar\alpha_t}x_t \\
\tilde\beta_t &= \frac{1 - \bar\alpha_{t-1}}{1 - \bar\alpha_t} \beta_t
\end{align}
The paper sets $\textcolor{lightgreen}{\Sigma_\theta}(x_t, t) = \sigma_t^2 \mathbf{I}$ where $\sigma_t^2$ is set to constants
$\beta_t$ or $\tilde\beta_t$.
Then,
$$\textcolor{lightgreen}{p_\theta}(x_{t-1} | x_t) = \mathcal{N}\big(x_{t-1}; \textcolor{lightgreen}{\mu_\theta}(x_t, t), \sigma_t^2 \mathbf{I} \big)$$
For given noise $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ using $q(x_t|x_0)$
\begin{align}
x_t(x_0, \epsilon) &= \sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\epsilon \\
x_0 &= \frac{1}{\sqrt{\bar\alpha_t}} \Big(x_t(x_0, \epsilon) -  \sqrt{1-\bar\alpha_t}\epsilon\Big)
\end{align}
This gives,
\begin{align}
L_{t-1}
 &= D_{KL}(q(x_{t-1}|x_t,x_0) \Vert \textcolor{lightgreen}{p_\theta}(x_{t-1}|x_t)) \\
 &= \mathbb{E}_q \Bigg[ \frac{1}{2\sigma_t^2}
 \Big \Vert \tilde\mu(x_t, x_0) - \textcolor{lightgreen}{\mu_\theta}(x_t, t) \Big \Vert^2 \Bigg] \\
 &= \mathbb{E}_{x_0, \epsilon} \Bigg[ \frac{1}{2\sigma_t^2}
  \bigg\Vert \frac{1}{\sqrt{\alpha_t}} \Big(
  x_t(x_0, \epsilon) - \frac{\beta_t}{\sqrt{1 - \bar\alpha_t}} \epsilon
  \Big) - \textcolor{lightgreen}{\mu_\theta}(x_t(x_0, \epsilon), t) \bigg\Vert^2 \Bigg] \\
\end{align}
Re-parameterizing with a model to predict noise
\begin{align}
\textcolor{lightgreen}{\mu_\theta}(x_t, t) &= \tilde\mu \bigg(x_t,
  \frac{1}{\sqrt{\bar\alpha_t}} \Big(x_t -
   \sqrt{1-\bar\alpha_t}\textcolor{lightgreen}{\epsilon_\theta}(x_t, t) \Big) \bigg) \\
  &= \frac{1}{\sqrt{\alpha_t}} \Big(x_t -
  \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\textcolor{lightgreen}{\epsilon_\theta}(x_t, t) \Big)
\end{align}
where $\epsilon_\theta$ is a learned function that predicts $\epsilon$ given $(x_t, t)$.
This gives,
\begin{align}
L_{t-1}
&= \mathbb{E}_{x_0, \epsilon} \Bigg[ \frac{\beta_t^2}{2\sigma_t^2 \alpha_t (1 - \bar\alpha_t)}
  \Big\Vert
  \epsilon - \textcolor{lightgreen}{\epsilon_\theta}(\sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\epsilon, t)
  \Big\Vert^2 \Bigg]
\end{align}
That is, we are training to predict the noise.
























## Dataset

For training we used the [TinyImageNet](https://courses.cs.washington.edu/courses/cse599g1/19au/files/homework2.tar) dataset with size `64 x 64 x 3`.  The images are normalized using mean as ```[0.485, 0.456, 0.406]``` and standard deviation as ```[0.229, 0.224, 0.225]``` 



## Experiments
Our model is trained on NVIDIA GTX 1050 Ti GPU.




## Results





## References

[Diffusion Models](https://medium.com/@monadsblog/diffusion-models-4dbe58489a2f) <br />
[Diffusion Models Made Easy](https://towardsdatascience.com/diffusion-models-made-easy-8414298ce4da) <br />
[EINOPS](https://github.com/arogozhnikov/einops)




