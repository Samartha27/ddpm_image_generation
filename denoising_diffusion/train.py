import numpy as np
from pathlib import Path
from random import random
from collections import namedtuple
import torch
from tqdm.auto import tqdm
from torch.optim import Adam
from torchvision.utils import save_image
from utils import constants, helpers

import os

# from dataloader import dataloader
from model.unet import Unet
# from denoising_diffusion.model import forward_diffusion_sampling

from dataloader import tiny_image_net


from model.unet import Unet
from model.diffusion import Diffusion

from utils.helpers import num_to_groups

results_folder = Path("./results")
results_folder.mkdir(exist_ok = True)
save_and_sample_every = 1000


timesteps = 200
batch_size = 128
image_size = 64
channels = 3
DATA_FILEPATH = "data/tiny_imagenet/train.h5"

if "PYTORCH_DEVICE" in os.environ:
    device = os.environ["PYTORCH_DEVICE"]
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"

model = Unet(

    dim= constants.image_size,
    channels= constants.channels,
    dim_mults=(1, 2, 4, 8,)
)
model.to(device)

diffusion = Diffusion(timesteps= constants.timesteps, device=device)

optimizer = Adam(model.parameters(), lr=1e-3)

loader = tiny_image_net.get_loader(DATA_FILEPATH, batch_size= constants.batch_size)

for epoch in range(constants.epochs):
    for step, batch in enumerate(loader):
        optimizer.zero_grad()

        batch_size = batch.shape[0]
        batch = batch.to(device)

        # Algorithm 1 line 3: sample t uniformally for every example in the batch
        t = torch.randint(0, constants.timesteps, (batch_size,), device=device).long()

        loss = diffusion.p_losses(model, batch, t, loss_type="huber")

        if step % 100 == 0:
            print("Loss:", loss.item())

        loss.backward()
        optimizer.step()

        # save generated images
        if step != 0 and step % save_and_sample_every == 0:
            milestone = step // save_and_sample_every
            batches = num_to_groups(4, batch_size)
            all_images_list = list(map(lambda n: diffusion.sample(model, batch_size=n, channels=constants.channels), batches))
            all_images = torch.cat(all_images_list, dim=0)
            all_images = (all_images + 1) * 0.5
            save_image(all_images, str(results_folder / f'sample-{milestone}.png'), nrow = 6)

