import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision

def setup_writers(log_dir):
    return SummaryWriter(f"{log_dir}/fake"), SummaryWriter(f"{log_dir}/real")

def log_images(gen, real, fixed_noise, step, writer_fake, writer_real):
    with torch.no_grad():
        fake = gen(fixed_noise).cpu()
        real = real.cpu()
        img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
        img_grid_real = torchvision.utils.make_grid(real, normalize=True)

        writer_fake.add_image("MNIST Fake Images", img_grid_fake, global_step=step)
        writer_real.add_image("MNIST Real Images", img_grid_real, global_step=step)
