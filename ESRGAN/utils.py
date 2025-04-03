import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import transforms

# Gradient Penalty Function
def gradient_penalty(discriminator, real_images, fake_images, device="cuda"):
    """
    Computes the gradient penalty for the Wasserstein GAN.
    """
    batch_size, channels, height, width = real_images.size()
    epsilon = torch.randn(batch_size, 1, 1, 1).to(device)
    
    interpolated_images = epsilon * real_images + (1 - epsilon) * fake_images
    interpolated_images.requires_grad_(True)
    
    d_interpolated = discriminator(interpolated_images)
    
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated_images,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True,
    )[0]
    
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1) ** 2).mean()
    
    return gradient_penalty

# Checkpoint Saving Function
def save_checkpoint(model, optimizer, filename="checkpoint.pth"):
    """
    Save the model and optimizer state to the checkpoint file.
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

# Checkpoint Loading Function
def load_checkpoint(filename, model, optimizer, lr=1e-4):
    """
    Load a checkpoint file and restore model and optimizer state.
    """
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    print(f"Checkpoint loaded from {filename}")
    return model, optimizer

# Function to plot and save examples (real vs generated images)
def plot_examples(output_dir, generator, num_images=5, size=(128, 128), device="cuda"):
    """
    Generates and saves example images to a directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a batch of random noise for generator
    noise = torch.randn(num_images, 3, size[0], size[1]).to(device)
    
    with torch.no_grad():
        gen_images = generator(noise)
    
    # Convert to numpy for plotting and save
    grid = torchvision.utils.make_grid(gen_images, nrow=5, normalize=True)
    save_image(grid, os.path.join(output_dir, "generated_images.png"))

# Save example images during training to monitor progress
def save_generated_images(fake_images, epoch, batch_idx, output_dir="output", prefix="fake", size=(128, 128)):
    """
    Save generated images to disk.
    """
    os.makedirs(output_dir, exist_ok=True)
    save_image(fake_images, f"{output_dir}/{prefix}_epoch{epoch}_batch{batch_idx}.png", normalize=True)

