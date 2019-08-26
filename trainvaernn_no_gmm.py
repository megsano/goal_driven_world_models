""" Recurrent model training """
import argparse
from functools import partial
from os.path import join, exists
from os import mkdir
import torch
import torch.nn.functional as f
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from utils.misc import save_checkpoint
from utils.misc import ASIZE, LSIZE, RSIZE, RED_SIZE, SIZE
from utils.learning import EarlyStopping
## WARNING : THIS SHOULD BE REPLACED WITH PYTORCH 0.5
from utils.learning import ReduceLROnPlateau
import pickle
from collections import Counter

from data.loaders import RolloutSequenceDataset
from models.vae import VAE
from models.vaernn_no_gmm import VAERNN_NOGMM, obs_loss

parser = argparse.ArgumentParser("VAERNN training")
parser.add_argument('--logdir', type=str,
                    help="Where things are logged and models are loaded from.")
parser.add_argument('--noreload', action='store_true',
                    help="Do not reload if specified.")
parser.add_argument('--include_reward', action='store_true',
                    help="Add a reward modelisation term to the loss.")
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(231)

# constants
BSIZE = 16
SEQ_LEN = 32
epochs = 9

#Pickle dictionary
pd = {}
pd['train'] = {}
pd['test'] = {}
pd['train']['av_loss'] = []
pd['train']['av_mse_loss'] = []
pd['train']['av_common_mse_loss'] = []
pd['train']['av_rare_mse_loss'] = []
pd['test']['av_loss'] = []
pd['test']['av_mse_loss'] = []
pd['test']['av_common_mse_loss'] = []
pd['test']['av_rare_mse_loss'] = []

# Loading VAE
vaernn_no_gmm_dir = join(args.logdir, 'vaernn_no_gmm')
vaernn__no_gmm_file = join(vaernn_no_gmm_dir, 'best.tar')

if not exists(vaernn_no_gmm_dir):
    mkdir(vaernn_no_gmm_dir)

vaernn_nogmm = VAERNN_NOGMM(LSIZE, ASIZE, RSIZE, 5)
vaernn_nogmm.to(device)
optimizer = torch.optim.RMSprop(vaernn_nogmm.parameters(), lr=1e-3, alpha=.9)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
earlystopping = EarlyStopping('min', patience=30)

if exists(vaernn__no_gmm_file) and not args.noreload:
    vaernn_no_gmm_state = torch.load(vaernn__no_gmm_file)
    print("Loading VAERNN_NO_GMM at epoch {} "
          "with test error {}".format(
              vaernn_no_gmm_state["epoch"], vaernn_no_gmm_state["precision"]))
    vaernn_nogmm.load_state_dict(vaernn_no_gmm_state["state_dict"])
    optimizer.load_state_dict(vaernn_no_gmm_state["optimizer"])
    scheduler.load_state_dict(vaernn_no_gmm_state['scheduler'])
    earlystopping.load_state_dict(vaernn_no_gmm_state['earlystopping'])
else:
    reload_file = './best.tar'
    state = torch.load(reload_file, map_location=lambda storage, loc:storage)
    vaernn_nogmm.vae.load_state_dict(state['state_dict'], strict=False)


# Data Loading
transform = transforms.Lambda(
    lambda x: np.transpose(x, (0, 3, 1, 2)) / 255)
train_loader = DataLoader(
    RolloutSequenceDataset('/home/gengar888/world-models/rollouts/', SEQ_LEN, transform, buffer_size=30),
    batch_size=BSIZE, num_workers=8, shuffle=True)
test_loader = DataLoader(
    RolloutSequenceDataset('/home/gengar888/world-models/rollouts/', SEQ_LEN, transform, train=False, buffer_size=10),
    batch_size=BSIZE, num_workers=8)

def get_loss(gt_obs, action, reward, terminal,
             gt_next_obs, include_reward: bool):
    """ Compute losses.

    The loss that is computed is:
    (GMMLoss(latent_next_obs, GMMPredicted) + MSE(reward, predicted_reward) +
         BCE(terminal, logit_terminal)) / (LSIZE + 2)
    The LSIZE + 2 factor is here to counteract the fact that the GMMLoss scales
    approximately linearily with LSIZE. All losses are averaged both on the
    batch and the sequence dimensions (the two first dimensions).

    :args latent_obs: (BSIZE, SEQ_LEN, LSIZE) torch tensor
    :args action: (BSIZE, SEQ_LEN, ASIZE) torch tensor
    :args reward: (BSIZE, SEQ_LEN) torch tensor
    :args latent_next_obs: (BSIZE, SEQ_LEN, LSIZE) torch tensor

    :returns: dictionary of losses, containing the gmm, the mse, the bce and
        the averaged loss.
    """
    gt_obs, action,\
        reward, terminal,\
        gt_next_obs = [arr.transpose(1, 0)
                           for arr in [gt_obs, action,
                                       reward, terminal,
                                       gt_next_obs]]

    recon_vae_obs, latents, rs, ds, recon_batch, gt_obs_mu, gt_obs_logsigma, mus, sigmas = vaernn_nogmm(action, gt_obs)
    # Reconstruction loss comparing predicted next observation to actual next observation
    obs_l = obs_loss(gt_next_obs, recon_vae_obs)

    gt_obs = gt_obs.transpose(1,0)
    gt_obs = f.upsample(gt_obs.view(-1, 3, SIZE, SIZE), size=RED_SIZE, mode='bilinear', align_corners=True)

    # Reconstruction loss for initial VAE encodings
    R = f.mse_loss(recon_batch, gt_obs, size_average=False)

    # KLD loss for latent states
    KLD_latent = -0.5 * torch.sum(1 + 2 * sigmas - mus.pow(2) - (2 * sigmas).exp())
    KLD = -0.5 * torch.sum(1 + 2 * gt_obs_logsigma - gt_obs_mu.pow(2) - (2 * gt_obs_logsigma).exp())

    bce = f.binary_cross_entropy_with_logits(ds, terminal)

    scale = 2 * 3 * SIZE * SIZE + 2 * LSIZE

    if include_reward:
        mask_common = (reward < 0) & (reward > -2)
        mask_rare = 1 - mask_common
        #print(mask_common, mask_rare)
        rare_r_pred, rare_r_true = rs[mask_rare], reward[mask_rare]
        common_r_pred,  common_r_true = rs[mask_common], reward[mask_common]

        rare_num = float(list(rare_r_pred.size())[0])
        common_num = float(list(common_r_pred.size())[0])

        #mse = f.mse_loss(rs, reward)
        common_mse = f.mse_loss(common_r_pred, common_r_true)
        rare_mse = f.mse_loss(rare_r_pred, rare_r_true)

        mse = common_mse
        if rare_num > 0.0:
            mse = (common_mse * 1/common_num) + (rare_mse * 1/rare_num)
        #print(common_mse, rare_mse, common_num, rare_num, mse)
        #print(common_mse, rare_mse, mse)

        scale += 2
    else:
        mse = 0
        scale += 1

    loss = (obs_l + bce + mse + R + KLD + KLD_latent) / scale

    return dict(obs_l=obs_l, bce=bce, mse=mse, R=R, KLD=KLD, KLD_latent=KLD_latent, loss=loss), common_mse, rare_mse


def data_pass(epoch, train, include_reward): # pylint: disable=too-many-locals
    """ One pass through the data """
    if train:
        vaernn_nogmm.train()
        loader = train_loader
    else:
        vaernn_nogmm.eval()
        loader = test_loader

    loader.dataset.load_next_buffer()

    y_true = np.array([])
    y_pred = np.array([])

    cum_loss = 0
    cum_obs = 0
    cum_bce = 0
    cum_mse = 0
    cum_mse_common = 0
    cum_mse_rare = 0
    cum_R = 0
    cum_KLD = 0
    cum_KLD_latent = 0

    pbar = tqdm(total=len(loader.dataset), desc="Epoch {}".format(epoch))
    for i, data in enumerate(loader):
        gt_obs, action, reward, terminal, gt_next_obs = [arr.to(device) for arr in data]
        #Batch size is 16 and seq_len is 32
        #gt_obs and gt_next_obs are both size (BSIZE, SEQ_LEN, ASIZE, SIZE, SIZE)

        if train:
            losses, common_mse, rare_mse = get_loss(gt_obs, action, reward,
                              terminal, gt_next_obs, include_reward)

            optimizer.zero_grad()
            losses['loss'].backward()
            optimizer.step()
        else:
            with torch.no_grad():
                losses, common_mse, rare_mse = get_loss(gt_obs, action, reward,
                              terminal, gt_next_obs, include_reward)

        cum_loss += losses['loss'].item()
        cum_obs += losses['obs_l'].item()
        cum_bce += losses['bce'].item()
        cum_mse += losses['mse'].item() if hasattr(losses['mse'], 'item') else \
            losses['mse']
        cum_mse_common += common_mse.item()
        cum_mse_rare += rare_mse.item()
        cum_R += losses['R'].item()
        cum_KLD += losses['KLD'].item()
        cum_KLD_latent += losses['KLD_latent'].item()

        pbar.set_postfix_str("loss={loss:10.6f} bce={bce:10.6f} "
                             "obs_l={obs_l:10.6f} mse={mse:10.6f} R={R:10.6f} KLD={KLD:10.6f} KLD_latent={KLD_latent:10.6f}".format(
                                 loss=cum_loss / (i + 1), bce=cum_bce / (i + 1),
                                 obs_l=cum_obs / (3 * SIZE * SIZE) / (i + 1), mse=cum_mse / (i + 1),
                                 R=cum_R / (3 * SIZE * SIZE)/ (i + 1), KLD=cum_KLD / LSIZE / (i + 1),
                                 KLD_latent=cum_KLD_latent / LSIZE / (i + 1)))
        pbar.update(BSIZE)
    pbar.close()
    if train:
        pd['train']['av_loss'].append(cum_loss * BSIZE / len(loader.dataset))
        pd['train']['av_mse_loss'].append(cum_mse * BSIZE / len(loader.dataset))
        pd['train']['av_common_mse_loss'].append(cum_mse_common * BSIZE / len(loader.dataset))
        pd['train']['av_rare_mse_loss'].append(cum_mse_rare * BSIZE / len(loader.dataset))
    else:
        pd['test']['av_loss'].append(cum_loss * BSIZE / len(loader.dataset))
        pd['test']['av_mse_loss'].append(cum_mse * BSIZE / len(loader.dataset))
        pd['test']['av_common_mse_loss'].append(cum_mse_common * BSIZE / len(loader.dataset))
        pd['test']['av_rare_mse_loss'].append(cum_mse_rare * BSIZE / len(loader.dataset))

    return cum_loss * BSIZE / len(loader.dataset)

train = partial(data_pass, train=True, include_reward=True)
test = partial(data_pass, train=False, include_reward=True)

cur_best = None
for e in range(1,epochs):
    train(e)
    test_loss = test(e)
    scheduler.step(test_loss)
    earlystopping.step(test_loss)

    is_best = not cur_best or test_loss < cur_best
    if is_best:
        cur_best = test_loss
    checkpoint_fname = join(vaernn_no_gmm_dir, 'checkpoint.tar')
    save_checkpoint({
        "state_dict": vaernn_nogmm.state_dict(),
        "optimizer": optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'earlystopping': earlystopping.state_dict(),
        "precision": test_loss,
        "epoch": e}, is_best, checkpoint_fname,
                    vaernn__no_gmm_file)

    if earlystopping.stop:
        print("End of Training because of early stopping at epoch {}".format(e))
        pickle.dump(pd, open('vae_rnn_no_gmm.p', 'wb'))
        break

pickle.dump(pd, open('vae_rnn_no_gmm.p', 'wb'))
