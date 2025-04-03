import torch
import config
from torch import nn, optim
from torch.utils.data import DataLoader
from model import Generator, Discriminator, initialize_weights
from loss import CustomLoss
from utils import save_checkpoint, load_checkpoint, plot_examples
from dataset import MyImageFolder
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def train_fn(
    loader,
    disc,
    gen,
    opt_gen,
    opt_disc,
    l1_loss,
    custom_loss,
    g_scaler,
    d_scaler,
    writer,
    tb_step,
):
    loop = tqdm(loader, leave=True)

    for idx, (low_res, high_res) in enumerate(loop):
        high_res = high_res.to(config.DEVICE)
        low_res = low_res.to(config.DEVICE)

        # Train the Discriminator
        with torch.cuda.amp.autocast():
            fake = gen(low_res)
            critic_real = disc(high_res)
            critic_fake = disc(fake.detach())
            gp = gradient_penalty(disc, high_res, fake, device=config.DEVICE)
            loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake)) + config.LAMBDA_GP * gp

        opt_disc.zero_grad()
        d_scaler.scale(loss_critic).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train the Generator
        with torch.cuda.amp.autocast():
            l1_loss_value = 1e-2 * l1_loss(fake, high_res)
            adversarial_loss = 5e-3 * -torch.mean(disc(fake))
            perceptual_loss = custom_loss(fake, high_res)
            gen_loss = l1_loss_value + perceptual_loss + adversarial_loss

        opt_gen.zero_grad()
        g_scaler.scale(gen_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        writer.add_scalar("Critic loss", loss_critic.item(), global_step=tb_step)
        writer.add_scalar("Generator loss", gen_loss.item(), global_step=tb_step)
        tb_step += 1

        if idx % 100 == 0 and idx > 0:
            plot_examples("test_images/", gen)

        loop.set_postfix(
            gp=gp.item(),
            critic=loss_critic.item(),
            l1=l1_loss_value.item(),
            perceptual=perceptual_loss.item(),
            adversarial=adversarial_loss.item(),
        )

    return tb_step


def main():
    # Set up dataset and dataloader
    dataset = MyImageFolder(root_dir="data/")
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        num_workers=config.NUM_WORKERS,
    )

    # Initialize generator and discriminator
    gen = Generator(in_channels=3).to(config.DEVICE)
    disc = Discriminator(in_channels=3).to(config.DEVICE)

    # Initialize weights
    initialize_weights(gen)
    initialize_weights(disc)

    # Optimizers
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.9))
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0
