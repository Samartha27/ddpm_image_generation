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
from model.unet import Unet
from dataloader import datasets
from model.diffusion import Diffusion





if __name__ == '__main__':


    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    loader = datasets.get_loader(constants.TRAIN_DIR, batch_size= constants.batch_size)

    model = Unet(
        dim = constants.image_size,
        channels = constants.channels,
        dim_mults = constants.dim_mults
        )
    model.to(device)

    optimizer = Adam(model.parameters(), lr = constants.l_rate)

    diffusion = Diffusion(timesteps = constants.timesteps, device=device)

    if not os.path.exists(constants.RESULTS_DIR):
        os.makedirs(constants.RESULTS_DIR)



    print("Training ......")
    for epoch in range(constants.epochs):
        for step, batch in enumerate(loader):
            optimizer.zero_grad()

            batch_size = batch.shape[0]
            batch = batch.to(device)

            # Sampling t uniformally for every example in the batch
            t = torch.randint(0, constants.timesteps, (batch_size,), device=device).long()

            loss = diffusion.p_losses(model, batch, t, loss_type="huber")

            if step % 100 == 0:
                print("Loss:", loss.item())

            loss.backward()
            optimizer.step()

            # save  images
            if step != 0 and step % constants.save_and_sample_every == 0:
                milestone = step // constants.save_and_sample_every
                batches = helpers.num_to_groups(4, batch_size)
                all_images_list = list(map(lambda n: diffusion.sample(model, batch_size=n, channels=constants.channels), batches))
                all_images = torch.cat(all_images_list, dim=0)
                all_images = (all_images + 1) * 0.5
                save_image(all_images, str(constants.RESULTS_DIR / f'sample-{milestone}.png'), nrow = 6)

