""" Training VAE """
import argparse
from os.path import join, exists
from os import mkdir
import numpy as np
import pandas as pd 

import torch
import torch.utils.data
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.utils import save_image

import pickle 

from sklearn.metrics import precision_recall_fscore_support
from collections import Counter 

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
parser.add_argument('--epochs', type=int, default=20, metavar='N',
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

train_loss_list = []
test_loss_list = []
train_total_prfs_dict_list = []
test_total_prfs_dict_list = []
train_y_pred_counts_list = []
test_y_pred_counts_list = []
train_y_true_counts_list = []
test_y_true_counts_list = []

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logsigma, predicted_reward, actual_reward):
    """ VAE loss function """
    BCE = F.mse_loss(recon_x, x, size_average=False)

    KLD = -0.5 * torch.sum(1 + 2 * logsigma - mu.pow(2) - (2 * logsigma).exp())

    actuals = []
    for i, act in enumerate(actual_reward): 
        if act.item() > -50 and act.item() < 0:
            actual = 0 # torch.tensor(0)#torch.tensor([1, 0, 0])
        elif act.item() <= -50:
            actual = 1 # torch.tensor(1)#torch.tensor([0, 1, 0])
        else:
            actual = 2 #torch.tensor(2)#torch.tensor([0, 0, 1])
        actuals.append(actual)
        
    actuals_tensor = torch.tensor(actuals)
    predicted_reward = predicted_reward.to(device) 
    actuals_tensor = actuals_tensor.to(device)
    
    RPL = F.cross_entropy(predicted_reward, actuals_tensor)
    predicted_reward_indices = torch.argmax(predicted_reward,1).data.cpu().numpy()
    
    return BCE + KLD + RPL, actuals, predicted_reward_indices


def train(epoch):
    """ One training epoch """
    model.train()
    dataset_train.load_next_buffer()
    train_loss = 0
    
#     default_reward = {'true_pos':0, 'true_neg':0, 'false_pos':0, 'false_neg':0}
#     offtrack_reward = {'true_pos':0, 'true_neg':0, 'false_pos':0, 'false_neg':0}
#     else_reward = {'true_pos':0, 'true_neg':0, 'false_pos':0, 'false_neg':0}
    
    total_prfs_dict = {'macro': (0, 0, 0), 'micro': (0, 0, 0), 'weighted':(0, 0, 0), 'none':(0, 0, 0)}
    y_true = np.array([])
    y_pred = np.array([])

    for batch_idx, data in enumerate(train_loader):
        obs, reward = data
        obs = obs.to(device)
        reward = reward.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar, predicted_reward = model(obs)
        actual_reward = reward
        loss, actuals, predicted_reward_indices = loss_function(recon_batch, obs, mu, logvar, predicted_reward, actual_reward)
        
#         total_prfs_dict['macro'] += prfs['macro']
#         total_prfs_dict['micro'] += prfs['micro']
#         total_prfs_dict['weighted'] += prfs['weighted']

        y_true = np.append(y_true, actuals) 
        y_pred = np.append(y_pred, predicted_reward_indices) 
        
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 20 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(obs), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(obs)))
            
#     total_prfs_dict['macro'] /= len(train_loader.dataset)
#     total_prfs_dict['micro'] /= len(train_loader.dataset)
#     total_prfs_dict['weighted'] /= len(train_loader.dataset)

#     default_p = float(default_reward['true_pos']) / float(default_reward['true_pos'] + default_reward['false_pos']) 
#     offtrack_p = float(offtract_reward['true_pos']) / float(offtract_reward['true_pos'] + offtrack_reward['false_pos']) 
#     else_p = float(else_reward['true_pos']) / float(else_reward['true_pos'] + else_reward['false_pos']) 
    
#     default_r = float(default_reward['true_pos']) / float(default_reward['true_pos'] + default_reward['false_neg']) 
#     offtrack_p = float(offtract_reward['true_pos']) / float(offtract_reward['true_pos'] + offtrack_reward['false_neg']) 
#     else_p = float(else_reward['true_pos']) / float(else_reward['true_pos'] + else_reward['false_neg']) 
    
#     default_f1 = 2 * default_p * default_r / (default_p + default_r) 
#     offtrack_f1 = 2 * offtrack_p * offtrack_r / (offtrack_p + offtrack_r) 
#     else_f1 = 2 * else_p * else_r / (else_p + else_r) 

    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    
    y_pred_counts = Counter(y_pred)
    y_true_counts = Counter(y_true)
    
    for avr in ['macro', 'micro', 'weighted']:
        total_prfs_dict[avr] = precision_recall_fscore_support(y_true, y_pred, average=avr)
    
    total_prfs_dict['none'] = precision_recall_fscore_support(y_true, y_pred, average=None)
    
    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))
    
    print('====> Epoch: {} Average macro: {} Average micro: {}  Average weighted: {}'.format(
        epoch, total_prfs_dict['macro'], total_prfs_dict['micro'], total_prfs_dict['weighted']))
    
    print(total_prfs_dict['none'])
    
    print("prediction counts: {}".format(y_pred_counts))
    print("actual counts: {}".format(y_true_counts))
    
    train_loss_list.append(train_loss / len(train_loader.dataset))
    train_total_prfs_dict_list.append(total_prfs_dict)
    train_y_pred_counts_list.append(y_pred_counts)
    train_y_true_counts_list.append(y_true_counts)
   


def test():
    """ One test epoch """
#     default_reward = {'true_pos':0, 'true_neg':0, 'false_pos':0, 'false_neg':0}
#     offtrack_reward = {'true_pos':0, 'true_neg':0, 'false_pos':0, 'false_neg':0}
#     else_reward = {'true_pos':0, 'true_neg':0, 'false_pos':0, 'false_neg':0}
    
    model.eval()
    dataset_test.load_next_buffer()
    test_loss = 0
    
    total_prfs_dict = {'macro': (0, 0, 0), 'micro': (0, 0, 0), 'weighted':(0, 0, 0), 'none':(0, 0, 0)}
    y_true = np.array([])
    y_pred = np.array([])
    
    with torch.no_grad():
        for data in test_loader:
            obs, reward = data
            obs= obs.to(device)
            reward=reward.to(device)
            actual_reward=reward
            recon_batch, mu, logvar, predicted_reward = model(obs)
            loss, actuals, predicted_reward_indices  = loss_function(recon_batch, obs, mu, logvar, predicted_reward, actual_reward)
            test_loss += loss.item()
            
            y_true = np.append(y_true, actuals) 
            y_pred = np.append(y_pred, predicted_reward_indices) 
            
#             total_prfs_dict['macro'] += prfs['macro']
#             total_prfs_dict['micro'] += prfs['micro']
#             total_prfs_dict['weighted'] += prfs['weighted']
            
    test_loss /= len(test_loader.dataset)
    
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    
    y_pred_counts = Counter(y_pred)
    y_true_counts = Counter(y_true)
    
    for avr in ['macro', 'micro', 'weighted']:
        total_prfs_dict[avr] = precision_recall_fscore_support(y_true, y_pred, average=avr)
    
    total_prfs_dict['none'] = precision_recall_fscore_support(y_true, y_pred, average=None)
    
#     total_prfs_dict['macro'] /= len(test_loader.dataset)
#     total_prfs_dict['micro'] /=  len(test_loader.dataset)
#     total_prfs_dict['weighted'] /=  len(test_loader.dataset)
    
#     default_p = float(default_reward['true_pos']) / float(default_reward['true_pos'] + default_reward['false_pos']) 
#     offtrack_p = float(offtract_reward['true_pos']) / float(offtract_reward['true_pos'] + offtrack_reward['false_pos']) 
#     else_p = float(else_reward['true_pos']) / float(else_reward['true_pos'] + else_reward['false_pos']) 
    
#     default_r = float(default_reward['true_pos']) / float(default_reward['true_pos'] + default_reward['false_neg']) 
#     offtrack_p = float(offtract_reward['true_pos']) / float(offtract_reward['true_pos'] + offtrack_reward['false_neg']) 
#     else_p = float(else_reward['true_pos']) / float(else_reward['true_pos'] + else_reward['false_neg']) 
    
#     default_f1 = 2 * default_p * default_r / (default_p + default_r) 
#     offtrack_f1 = 2 * offtrack_p * offtrack_r / (offtrack_p + offtrack_r) 
#     else_f1 = 2 * else_p * else_r / (else_p + else_r) 
    
    
    print('====> Test set loss: {:.4f}'.format(test_loss))
    print('====> Epoch: {} Average macro: {} Average micro: {}  Average weighted: {}'.format(
        epoch, total_prfs_dict['macro'], total_prfs_dict['micro'], total_prfs_dict['weighted']))
    
    print(total_prfs_dict['none'])
    
    print("prediction counts: {}".format(y_pred_counts))
    print("actual counts: {}".format(y_true_counts))
    
    test_loss_list.append(test_loss / len(test_loader.dataset))
    test_total_prfs_dict_list.append(total_prfs_dict)
    test_y_pred_counts_list.append(y_pred_counts)
    test_y_true_counts_list.append(y_true_counts)
    
    return test_loss

# check vae dir exists, if not, create it
vae_dir = join(args.logdir, 'vae_reward_eval/')
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
        pickle.dump((train_loss_list, test_loss_list, train_total_prfs_dict_list, test_total_prfs_dict_list, train_y_pred_counts_list, test_y_pred_counts_list, train_y_true_counts_list, test_y_true_counts_list), open('vae_reward_eval_scores.p', 'wb'))
        break
        
pickle.dump((train_loss_list, test_loss_list, train_total_prfs_dict_list, test_total_prfs_dict_list, train_y_pred_counts_list, test_y_pred_counts_list, train_y_true_counts_list, test_y_true_counts_list), open('vae_reward_eval_scores.p', 'wb'))
