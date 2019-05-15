""" Training VAE """
import argparse
from os.path import join, exists
from os import mkdir
import numpy as np

import torch
import torch.utils.data
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.utils import save_image

from models.vae_reward import VAE

from utils.misc import save_checkpoint
from utils.misc import LSIZE, RED_SIZE
## WARNING : THIS SHOULD BE REPLACE WITH PYTORCH 0.5
from utils.learning import EarlyStopping
from utils.learning import ReduceLROnPlateau
from data.loaders import RolloutObservationDataset
from data.loaders import RolloutSequenceDataset
from data.loaders import RolloutRewardDataset

parser = argparse.ArgumentParser(description='VAE Trainer')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='number of epochs to train (default: 1000)')
parser.add_argument('--logdir', type=str, help='Directory where results are logged')
parser.add_argument('--noreload', action='store_true',
                    help='Best model is not reloaded if specified')
parser.add_argument('--nosamples', action='store_true',
                    help='Does not save samples during training if specified')


args = parser.parse_args()
cuda = torch.cuda.is_available()


torch.manual_seed(123)
# Fix numeric divergence due to bug in Cudnn
torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if cuda else "cpu")

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((RED_SIZE, RED_SIZE)),
    transforms.ToTensor()
])

SEQ_LEN = 32 
BSIZE=32

dataset_train = RolloutRewardDataset('/home/gengar888/world-models/rollouts/', 
                                          transform, train=True)
dataset_test = RolloutRewardDataset('/home/gengar888/world-models/rollouts/', 
                                         transform, train=False)

train_loader = torch.utils.data.DataLoader(
    dataset_train, batch_size=32, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(
    dataset_test, batch_size=32, shuffle=True, num_workers=0)

model = VAE(3, LSIZE).to(device)
optimizer = optim.Adam(model.parameters())
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
earlystopping = EarlyStopping('min', patience=30)

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logsigma, predicted_reward, actual_reward):
    """ VAE loss function """
    BCE = F.mse_loss(recon_x, x, size_average=False)

    KLD = -0.5 * torch.sum(1 + 2 * logsigma - mu.pow(2) - (2 * logsigma).exp())
    
#     if torch.argmax(predicted_reward) == 0:
#         RPL = 0 if actual_reward == -0.1 else 1 
#     elif torch.argmax(predicted_reward) == 1:
#         RPL = 0 if actual_reward == -100 else 1 
#     else:
#         RPL = 0 if actual_reward != -0.1 and actual_reward != -100 else 1 
    
    actuals = []
    for act in actual_reward: 
        if act == -0.1:
            actual = 0 # torch.tensor(0)#torch.tensor([1, 0, 0])
        elif act == -100:
            actual = 1 # torch.tensor(1)#torch.tensor([0, 1, 0])
        else:
            actual = 2 #torch.tensor(2)#torch.tensor([0, 0, 1])
        actuals.append(actual)
    actuals = torch.tensor(actuals)
    actuals = actuals.to(device)

    RPL = F.cross_entropy(predicted_reward, actuals)
    
    return BCE + KLD + RPL 


def train(epoch):
    """ One training epoch """
    model.train()
    dataset_train.load_next_buffer()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        obs, reward = data
        obs = obs.to(device)
        reward = reward.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar, predicted_reward = model(obs)
        actual_reward = reward
        loss = loss_function(recon_batch, obs, mu, logvar, predicted_reward, actual_reward)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 20 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(obs), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(obs)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))


def test():
    """ One test epoch """
    model.eval()
    dataset_test.load_next_buffer()
    test_loss = 0
    with torch.no_grad():
        for data in test_loader:
            obs, reward = data
            obs= obs.to(device)
            reward=reward.to(device)
            actual_reward=reward
            recon_batch, mu, logvar, predicted_reward = model(obs)
            test_loss += loss_function(recon_batch, obs, mu, logvar, predicted_reward, actual_reward).item()

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss

# check vae dir exists, if not, create it
vae_dir = join(args.logdir, 'vae_reward/')
if not exists(vae_dir):
    mkdir(vae_dir)
    mkdir(join(vae_dir, 'samples'))

reload_file = join(vae_dir, 'best.tar')
if not args.noreload and exists(reload_file):
    state = torch.load(reload_file)
    print("Reloading model at epoch {}"
          ", with test error {}".format(
              state['epoch'],
              state['precision']))
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    earlystopping.load_state_dict(state['earlystopping'])


cur_best = None

for epoch in range(1, args.epochs + 1):
    train(epoch)
    test_loss = test()
    scheduler.step(test_loss)
    earlystopping.step(test_loss)

    # checkpointing
    best_filename = join(vae_dir, 'best.tar')
    filename = join(vae_dir, 'checkpoint.tar')
    is_best = not cur_best or test_loss < cur_best
    if is_best:
        cur_best = test_loss

    save_checkpoint({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'precision': test_loss,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'earlystopping': earlystopping.state_dict()
    }, is_best, filename, best_filename)



    if not args.nosamples:
        with torch.no_grad():
            sample = torch.randn(RED_SIZE, LSIZE).to(device)
            sample = model.decoder(sample).cpu()
            save_image(sample.view(64, 3, RED_SIZE, RED_SIZE),
                       join(vae_dir, 'samples/sample_' + str(epoch) + '.png'))

    if earlystopping.stop:
        print("End of Training because of early stopping at epoch {}".format(epoch))
        break
