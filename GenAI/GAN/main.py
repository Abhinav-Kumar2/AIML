from config import config
from models.generator import Generator
from models.discriminator import Discriminator
from data_stuff.dataloader import data_loader
from train import train_gan
from utils.logging import setup_writers

def main():
    loader = data_loader(config["batch_size"], config["dataset_path"])

    gen = Generator(config["z_dim"]).to(config["device"])
    disc = Discriminator().to(config["device"])

    writer_fake, writer_real = setup_writers(config["log_dir"])

    train_gan(disc, gen, loader, config, writer_fake, writer_real)

if __name__ == "__main__":
    main()
