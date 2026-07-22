import os
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader

from gan_models import Generator, Discriminator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Generate Outputs

os.makedirs("output/generated_samples", exist_ok=True)



latent_dim = 100
batch_size = 128
learning_rate = 0.0002
epochs = 20

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# LOAD THE MNIST DATASET

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    transform=transform,
    download=True
)


# DATALOADER

dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)


generator = Generator(latent_dim).to(device)
discriminator = Discriminator().to(device)


criterion = nn.BCELoss()

optimizer_G = optim.Adam(
    generator.parameters(),
    lr=learning_rate,
    betas=(0.5, 0.999)
)

optimizer_D = optim.Adam(
    discriminator.parameters(),
    lr=learning_rate,
    betas=(0.5, 0.999)
)


# TRAINING LOOP

for epoch in range(epochs):

    for batch_idx, (real_images, _) in enumerate(dataloader):

        real_images = real_images.to(device)

        batch_size_current = real_images.size(0)

        real_labels = torch.ones(batch_size_current, 1).to(device)
        fake_labels = torch.zeros(batch_size_current, 1).to(device)


        # Discriminator Training Block

        optimizer_D.zero_grad()

        real_outputs = discriminator(real_images)
        real_loss = criterion(real_outputs, real_labels)

        noise = torch.randn(batch_size_current, latent_dim).to(device)
        fake_images = generator(noise)

        fake_outputs = discriminator(fake_images.detach())
        fake_loss = criterion(fake_outputs, fake_labels)

        d_loss = real_loss + fake_loss

        d_loss.backward()
        optimizer_D.step()


        # Generator Training Block

        optimizer_G.zero_grad()

        outputs = discriminator(fake_images)

        g_loss = criterion(outputs, real_labels)

        g_loss.backward()

        optimizer_G.step()


    print(
        f"Epoch [{epoch+1}/{epochs}] | "
        f"D Loss: {d_loss.item():.4f} | "
        f"G Loss: {g_loss.item():.4f}"
    )

    if (epoch + 1) % 5 == 0:
        with torch.no_grad():
            noise = torch.randn(64, latent_dim).to(device)

            generated_images = generator(noise)

            save_image(
                generated_images,
                f"output/generated_samples/epoch_{epoch+1:03d}.png",
                nrow=8,
                normalize=True
            )


torch.save(generator.state_dict(), "generator.pth")
torch.save(discriminator.state_dict(), "discriminator.pth")

print("GAN Training Completed Successfully!")
