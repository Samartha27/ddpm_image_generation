import numpy as np
import torch
from torch import nn, einsum
import torch.nn.functional as F
import matplotlib.animation as animation
import matplotlib as plt


image_size = 28
channels = 1
batch_size = 128
timesteps = 200

# sample 64 images
samples = sample(model, image_size=image_size, batch_size=64, channels=channels)


# show a random one
random_index = 5
plt.imshow(samples[-1][random_index].reshape(image_size, image_size, channels), cmap="gray")


random_index = 53

fig = plt.figure()
ims = []
for i in range(timesteps):
    im = plt.imshow(samples[i][random_index].reshape(image_size, image_size, channels), cmap="gray", animated=True)
    ims.append([im])

animate = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
animate.save('diffusion.gif')
plt.show()

