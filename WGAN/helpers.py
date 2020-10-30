from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision.utils import make_grid
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from loss import *
import torch

torch.manual_seed(0)


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=0, batch_norm=True, **kwargs):
        super(Conv, self).__init__()
        padding_mode = kwargs.get('padding_mode', 'zeros')
        self.batch_norm = batch_norm
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, padding_mode=padding_mode)
        if self.batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.batch_norm:
            x = self.bn(x)
        return x


class UpOrTrans(nn.Module):
    def __init__(self, in_channels, out_channels, mode='trans', kernel_size=1, stride=1, scale_factor=2, padding=0,
                 batch_norm=True, **kwargs):
        super(UpOrTrans, self).__init__()
        self.mode = mode
        upmode = kwargs.get('upmode', 'nearest')
        align_corners = kwargs.get('align_corners', None)
        self.batch_norm = batch_norm

        if self.mode == 'up':
            self.up = nn.Upsample(scale_factor=scale_factor, mode=upmode, align_corners=align_corners)
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

        if self.mode == 'trans':
            self.trans = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)

        if self.batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x, output_size=None):
        if self.mode == 'up':
            x = self.up(x)
            x = self.conv(x)
        if self.mode == 'trans':
            x = self.trans(x, output_size=output_size)

        if self.batch_norm:
            x = self.bn(x)
        return x


def build_dataloader(folder_path, batch, image_dim):
    """
    Builds dataloader from a folder path

    :param folder_path: folder where image consists
    :param batch: batch size
    :param image_dim: image dimension to take
    :return: data_loader: pytorch dataloader
    """
    transform = transforms.Compose([
        transforms.Resize((image_dim, image_dim)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    data = datasets.ImageFolder(folder_path, transform=transform)
    data_loader = DataLoader(data, batch_size=batch)
    return data_loader


def get_noise(n_samples, z_dim, device='cpu'):
    """
    Function for creating noise vectors: Given the dimensions (n_samples, z_dim)
    creates a tensor of that shape filled with random numbers from the normal distribution.

    :param n_samples: the number of samples to generate, a scalar
    :param z_dim: the dimension of the noise vector, a scalar
    :param device: the device type

    :return random values
    """
    return torch.randn(n_samples, z_dim, device=device)


def show_tensor_images(image_tensor, num_images=5, name=None):
    """
    Function for visualizing images: Given a tensor of images, number of images,
    plots and prints the images in an uniform grid.

    :param image_tensor: batch of image tensors
    :param num_images: number of images to show
    :param name: choose name if you want to save

    """
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    if name is not None:
        plt.savefig(str(name)+'.jpg')
    plt.show()


def train(dataloader,
          generator_model, critic_model,
          generator_optim, critic_optim,
          n_epochs, device='cuda',
          critic_repeat=1, gen_repeat=1,
          display_step=100, z_dim=64,
          c_lambda=10):
    """
    Trains the  model

    :param dataloader: DataLoader consisting of image tensors
    :param generator_model: Generator model
    :param critic_model:   Critic(discriminator) model
    :param generator_optim: Generator optimizer
    :param critic_optim:  Critic optimizer
    :param n_epochs:  No of epochs to run
    :param device:  Device type
    :param critic_repeat: Run the critic multiple times every time when running the generator
    :param gen_repeat: Run the generator multiple times every time when running the critic
    :param display_step: how often to display/visualize the images
    :param z_dim: Dimension of the noise vector
    :param c_lambda:  Weight of the gradient penalty

    :return: None (saves models named 'G.pth' , 'C.pth')
    """
    generator_losses = []
    critic_losses = []
    for epoch in range(n_epochs):
        for idx, i in enumerate(tqdm(dataloader)):
            mean_iteration_critic_loss = 0
            mean_iteration_gen_loss = 0
            real = i[0].to(device)
            cur_batch_size = len(real)

            for _ in range(critic_repeat):
                critic_optim.zero_grad()
                fake_noise = get_noise(cur_batch_size, z_dim, device=device)
                fake = generator_model(fake_noise)
                crit_fake_pred = critic_model(fake.detach())
                crit_real_pred = critic_model(real)
                epsilon = torch.rand(len(real), 3, 1, 1, device=device, requires_grad=True)

                gradient = get_gradient(critic_model, real, fake.detach(), epsilon)
                gp = gradient_penalty(gradient)
                crit_loss = get_crit_loss(crit_fake_pred, crit_real_pred, gp, c_lambda)
                mean_iteration_critic_loss += crit_loss.item()
                crit_loss.backward(retain_graph=True)
                critic_optim.step()
            critic_losses += [mean_iteration_critic_loss]

            for _ in range(gen_repeat):
                generator_optim.zero_grad()
                fake_noise_2 = get_noise(cur_batch_size, z_dim, device=device)
                fake_2 = generator_model(fake_noise_2)
                crit_fake_pred = critic_model(fake_2)
                gen_loss = get_gen_loss(crit_fake_pred)
                mean_iteration_gen_loss += gen_loss.item()
                gen_loss.backward()
                generator_optim.step()
            generator_losses += [mean_iteration_gen_loss]

            if idx % display_step == 0:
                show_tensor_images(fake)
                show_tensor_images(real)
        torch.save(generator_model.state_dict(), 'G.pth')
        torch.save(critic_model.state_dict(), 'C.pth')