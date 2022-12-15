import numpy as np
import copy
from pathlib import Path
from random import random
from collections import namedtuple
from multiprocessing import cpu_count
import torch
from torch import nn, einsum
import torch.nn.functional as F
from tqdm.auto import tqdm
from torch.optim import Adam
from torchvision.utils import save_image
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
from utils import constants
from dataloader import dataloader


results_folder = Path("./results")
results_folder.mkdir(exist_ok = True)
save_and_sample_every = 1000



batch = next(iter(dataloader))

device = "cuda" if torch.cuda.is_available() else "cpu"

model = Unet(
    dim= constants.image_size,
    channels= constants.channels,
    dim_mults=(1, 2, 4,)
)
model.to(device)

optimizer = Adam(model.parameters(), lr=1e-3)


for epoch in range(constants.epochs):
    for step, batch in enumerate(dataloader):
      optimizer.zero_grad()

      batch_size = batch["pixel_values"].shape[0]
      batch = batch["pixel_values"].to(device)

      # Algorithm 1 line 3: sample t uniformally for every example in the batch
      t = torch.randint(0, constants.timesteps, (batch_size,), device=device).long()

      loss = p_losses(model, batch, t, loss_type="huber")

      if step % 100 == 0:
        print("Loss:", loss.item())

      loss.backward()
      optimizer.step()

      # save generated images
      if step != 0 and step % save_and_sample_every == 0:
        milestone = step // save_and_sample_every
        batches = num_to_groups(4, batch_size)
        all_images_list = list(map(lambda n: sample(model, batch_size=n, channels= constants.channels), batches))
        all_images = torch.cat(all_images_list, dim=0)
        all_images = (all_images + 1) * 0.5
        save_image(all_images, str(results_folder / f'sample-{milestone}.png'), nrow = 6)