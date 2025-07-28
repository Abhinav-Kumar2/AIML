import torch
import torch.nn as nn
import torch.optim as optim
from utils.logging import log_images

def train_gan(disc, gen, loader, config, writer_fake, writer_real):
    opt_disc = optim.Adam(disc.parameters(), lr=config["lr"], betas=(0.5, 0.999))
    opt_gen = optim.Adam(gen.parameters(), lr=config["lr"], betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    fixed_noise = torch.randn((config["batch_size"], config["z_dim"], 1, 1)).to(config["device"])
    step = 0

    for epoch in range(config["num_epochs"]):
        for batch_idx, (real, _) in enumerate(loader):
            real = real.to(config["device"])
            batch_size = real.size(0)

            noise = torch.randn(batch_size, config["z_dim"], 1, 1).to(config["device"])
            fake = gen(noise)

            disc_real = disc(real).view(-1)
            lossD_real = criterion(disc_real, torch.ones_like(disc_real))
            disc_fake = disc(fake.detach()).view(-1)
            lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            lossD = (lossD_real + lossD_fake) / 2

            disc.zero_grad()
            lossD.backward()
            opt_disc.step()

            output = disc(fake).view(-1)
            lossG = criterion(output, torch.ones_like(output))

            gen.zero_grad()
            lossG.backward()
            opt_gen.step()

            if batch_idx == 0:
                print(
                    f"Epoch [{epoch}/{config['num_epochs']}] Batch {batch_idx}/{len(loader)} "
                    f"Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
                )
                log_images(gen, real, fixed_noise, step, writer_fake, writer_real)
                step += 1
