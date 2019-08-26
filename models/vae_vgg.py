
"""
Variational encoder model, used as a visual model
for our model of the world.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, models, transforms

class Decoder(nn.Module):
    """ Transfer learning VGG encoder """
    def __init__(self, img_channels, latent_size):
        super(Decoder, self).__init__()
        self.latent_size = latent_size
        self.img_channels = img_channels

        self.fc1 = nn.Linear(latent_size, 1024)
        self.deconv1 = nn.ConvTranspose2d(1024, 128, 5, stride=2)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
        self.deconv4 = nn.ConvTranspose2d(32, img_channels, 6, stride=2)


    def forward(self, x): # pylint: disable=arguments-differ
        x = F.relu(self.fc1(x))
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        reconstruction = F.sigmoid(self.deconv4(x))
        return reconstruction

class Encoder(nn.Module): # pylint: disable=too-many-instance-attributes
    """ Transfer learning VGG encoder """
    def __init__(self, img_channels, latent_size):
        super(Encoder, self).__init__()


        self.latent_size = latent_size
        self.img_channels = img_channels

        self.resize_conv = nn.ConvTranspose2d(img_channels, img_channels, 4, stride=4)

        self.model_conv = torchvision.models.vgg16(pretrained=True)
        for param in self.model_conv.parameters():
            param.requires_grad = False

        # Parameters of newly constructed modules have requires_grad=True by default
        num_ftrs = self.model_conv.classifier._modules['6'].in_features
        self.model_conv.classifier._modules['6'] = nn.Linear(num_ftrs, latent_size)

    def forward(self, x): # pylint: disable=arguments-differ
        x = self.model_conv(x)
        return x

class VAE_VGG(nn.Module):
    """ Variational Autoencoder """
    def __init__(self, img_channels, latent_size):
        super(VAE_VGG, self).__init__()
        self.encoder = Encoder(img_channels, latent_size)
        self.decoder = Decoder(img_channels, latent_size)

    def forward(self, x): # pylint: disable=arguments-differ
        z = self.encoder(x)
        recon_x = self.decoder(z)
        return recon_x, z
