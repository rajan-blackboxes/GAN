import torch.optim as optim
import torch
import sys
import torch
import os
import torch.nn as nn
import torch.nn.functional as F


def conv(in_channels, out_channels, kernel_size, stride=2,padding=1, batch_norm=True):
    layers = []
    conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
    layers.append(conv_layer)
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)
        
def deconv(in_channels, out_channels, kernel_size, stride=2,padding=1, batch_norm=True):
    layers = []
    conv_layer = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
    layers.append(conv_layer)
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)



######## models #######
class Discriminator(nn.Module):

    def __init__(self, conv_dim):
        """
        Initialize the Discriminator Module
        :param conv_dim: The depth of the first convolutional layer
        """
        super(Discriminator, self).__init__()
        self.conv_dim = conv_dim

        # complete init function
        # input: (32,32,3)
        self.conv1 = conv(3, conv_dim, 4, batch_norm=False) #(16,16,32)
        self.conv2 = conv(conv_dim, conv_dim*2, 4) #(8,8, 64)
        self.conv3 = conv(conv_dim*2, conv_dim*4, 4) #(4,4, 128)
        
        self.fc1 = nn.Linear(4*4*conv_dim*4, 1) #(2048,1)

    def forward(self, x):
        """
        Forward propagation of the neural network
        :param x: The input to the neural network     
        :return: Discriminator logits; the output of the neural network
        """
        # define feedforward behavior
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        
        x = x.view(-1, 4*4*self.conv_dim*4)
        
        x = self.fc1(x)
        return x


class Generator(nn.Module):
    
    def __init__(self, z_size, conv_dim):
        """
        Initialize the Generator Module
        :param z_size: The length of the input latent vector, z
        :param conv_dim: The depth of the inputs to the *last* transpose convolutional layer
        """
        super(Generator, self).__init__()
        self.conv_dim = conv_dim
        
        # complete init function
        self.fc = nn.Linear(z_size,4*4*(conv_dim*4)) #reshape to (z_size, 256,4,4)
        self.deconv1 = deconv(conv_dim*4, conv_dim*2, 4) # (4*4*128)
        self.deconv2 = deconv(conv_dim*2, conv_dim, 4) # (8*8*)
        self.deconv3 = deconv(conv_dim, 3, 4)
        self.deconv4_res = deconv(3, 3, 1, stride=1, padding=0)
        self.deconv5_res = deconv(3,3,1, stride=1, padding=0)
        self.deconv6 = deconv(3, 3, 1, stride=1, padding=0, batch_norm=False)
        self.dropout = nn.Dropout(0.5)
    def forward(self, x):
        """
        Forward propagation of the neural network
        :param x: The input to the neural network     
        :return: A 32x32x3 Tensor image as output
        """
        # define feedforward behavior
        x = self.fc(x) # (conv_dim, 2048)
        x = x.view(-1,self.conv_dim*4,4,4) #(128, 4,4)
        
        x = self.deconv1(x) # (64, 8, 8)
        self.dropout(x)
        x = F.relu(x)
        
        x = self.deconv2(x)  # (32,16,16)
        x = F.relu(x)
        self.dropout(x)
        
        x = self.deconv3(x) #(3, 32,32)
        x = x+F.relu(x)
        self.dropout(x)
        
        x = self.deconv4_res(x) # (3,32,32)
        x = x+F.relu(x)
        self.dropout(x)
        
        x = x + self.deconv5_res(x) #(3,32,32)
        self.dropout(x)
        
        x = x + self.deconv6(x) #(3,32,32)
#         self.dropout(x)
        out = torch.tanh(x) 
        
        return out




def weights_init_normal(m):
    """
    Applies initial weights to certain layers in a model .
    The weights are taken from a normal distribution 
    with mean = 0, std dev = 0.02.
    :param m: A module or layer in a network    
    """
    classname = m.__class__.__name__
    lr = 0.01
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if m.bias is not None:
                m.bias.data.fill_(0)



def build_network(d_conv_dim, g_conv_dim, z_size):
    # define discriminator and generator
    D = Discriminator(d_conv_dim)
    G = Generator(z_size=z_size, conv_dim=g_conv_dim)

    # initialize model weights
    D.apply(weights_init_normal)
    G.apply(weights_init_normal)
    return D, G




# Define model hyperparams
d_conv_dim = 32
g_conv_dim = 32
z_size = 100

D, G = build_network(d_conv_dim, g_conv_dim, z_size)
