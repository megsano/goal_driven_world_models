"""
Define the VAE RNN .
"""
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.distributions.normal import Normal
from models.vae import VAE
from utils.misc import ASIZE, LSIZE, RSIZE, RED_SIZE, SIZE

def gmm_loss(gt_next_obs, recon_next_obs, sigma_obs, logpi, reduce=True): # pylint: disable=too-many-arguments
    """ Computes the gmm loss.

    Compute minus the log probability of batch under the GMM model described
    by mus, sigmas, pi. Precisely, with bs1, bs2, ... the sizes of the batch
    dimensions (several batch dimension are useful when you have both a batch
    axis and a time step axis), gs the number of mixtures and fs the number of
    features.

    :args batch: (bs1, bs2, *, fs) torch tensor
    :args mus: (bs1, bs2, *, gs, fs) torch tensor
    :args sigmas: (bs1, bs2, *, gs, fs) torch tensor
    :args logpi: (bs1, bs2, *, gs) torch tensor
    :args reduce: if not reduce, the mean in the following formula is ommited

    :returns:
    loss(batch) = - mean_{i1=0..bs1, i2=0..bs2, ...} log(
        sum_{k=1..gs} pi[i1, i2, ..., k] * N(
            batch[i1, i2, ..., :] | mus[i1, i2, ..., k, :], sigmas[i1, i2, ..., k, :]))

    NOTE: The loss is not reduced along the feature dimension (i.e. it should scale ~linearily
    with fs).
    """
    # gt_next_obs = gt_next_obs.transpose(1,0)
    # gt_next_obs = f.upsample(gt_next_obs.view(-1, 3, SIZE, SIZE), size=RED_SIZE, mode='bilinear', align_corners=True)
    # gt_next_obs = gt_next_obs.view(recon_next_obs.size(0), recon_next_obs.size(1), recon_next_obs.size(3))
    # gt_next_obs = gt_next_obs.unsqueeze(-2)
    # print(gt_next_obs.size(), recon_next_obs.size(), sigma_obs.size())
    # normal_dist = Normal(recon_next_obs, sigma_obs)
    # g_log_probs = normal_dist.log_prob(gt_next_obs)
    # g_log_probs = logpi + torch.sum(g_log_probs, dim=-1)
    # max_log_probs = torch.max(g_log_probs, dim=-1, keepdim=True)[0]
    # g_log_probs = g_log_probs - max_log_probs
    #
    # g_probs = torch.exp(g_log_probs)
    # probs = torch.sum(g_probs, dim=-1)
    #
    # log_prob = max_log_probs.squeeze() + torch.log(probs)
    # if reduce:
    #     return - torch.mean(log_prob)
    # return - log_prob

    # this is where jank file starts

    gt_next_obs = gt_next_obs.transpose(1,0)
    gt_next_obs = f.upsample(gt_next_obs.view(-1, 3, SIZE, SIZE), size=RED_SIZE, mode='bilinear', align_corners=True)
    gt_next_obs = gt_next_obs.view(recon_next_obs.size(0), recon_next_obs.size(1), recon_next_obs.size(3))
    recon_next_obs = torch.mean(recon_next_obs, 2)
    BCE = f.mse_loss(gt_next_obs, recon_next_obs, size_average=False)
    # gt_next_obs = gt_next_obs.unsqueeze(-2)
    # print(gt_next_obs.size(), recon_next_obs.size(), sigma_obs.size())
    # normal_dist = Normal(recon_next_obs, sigma_obs)
    # g_log_probs = normal_dist.log_prob(gt_next_obs)
    # g_log_probs = logpi + torch.sum(g_log_probs, dim=-1)
    # max_log_probs = torch.max(g_log_probs, dim=-1, keepdim=True)[0]
    # g_log_probs = g_log_probs - max_log_probs
    #
    # g_probs = torch.exp(g_log_probs)
    # probs = torch.sum(g_probs, dim=-1)
    #
    # log_prob = max_log_probs.squeeze() + torch.log(probs)
    # if reduce:
    #     return - torch.mean(log_prob)
    # return - log_prob

    return BCE

class _VAERNNBase(nn.Module):
    def __init__(self, latents, actions, hiddens, gaussians):
        super().__init__()
        self.latents = latents
        self.actions = actions
        self.hiddens = hiddens
        self.gaussians = gaussians

        self.gmm_linear = nn.Linear(
            hiddens, (2 * latents + 1) * gaussians + 2)

    def forward(self, *inputs):
        pass

class VAERNN(_VAERNNBase):
    """ VAERNN model for multi steps forward """
    def __init__(self, latents, actions, hiddens, gaussians):
        super().__init__(latents, actions, hiddens, gaussians)
        self.rnn = nn.LSTM(latents + actions, hiddens)
        self.vae = VAE(3, latents)# .to(device)

    def forward(self, actions, gt_obs): # pylint: disable=arguments-differ ### change to obs
        """ MULTI STEPS forward.

        :args actions: (SEQ_LEN, BSIZE, ASIZE) torch tensor
        :args obs: (SEQ_LEN, BSIZE, OSIZE1, OSIZE2) torch tensor

        :returns: mu_nlat, sig_nlat, pi_nlat, rs, ds, parameters of the GMM
        prediction for the next latent, gaussian prediction of the reward and
        logit prediction of terminality.
            - mu_nlat: (SEQ_LEN, BSIZE, N_GAUSS,  LSIZE) torch tensor
            - sigma_nlat: (SEQ_LEN, BSIZE, N_GAUSS,  LSIZE) torch tensor
            - logpi_nlat: (SEQ_LEN, BSIZE, N_GAUSS) torch tensor
            - rs: (SEQ_LEN, BSIZE) torch tensor
            - ds: (SEQ_LEN, BSIZE) torch tensor
        """
        SEQ_LEN, BSIZE = actions.size(0), actions.size(1) # batch of actions for each time step in the sequence (t steps) T x N  x A
        bs, seq_len = BSIZE, SEQ_LEN

        #Turn 5d obs into 4d latent vector

        # obs = [f.upsample(x.view(-1, 3, SIZE, SIZE), size=RED_SIZE, mode='bilinear', align_corners=True) for x in obs]
        #
        # (obs_mu, obs_logsigma), _ = [self.vae(x)[1:] for x in (obs,)]
        #
        # latents = (obs_mu + obs_logsigma.exp() * torch.randn_like(obs_mu)).view(BSIZE, SEQ_LEN, LSIZE)

        gt_obs = gt_obs.transpose(1,0)
        gt_obs = f.upsample(gt_obs.view(-1, 3, SIZE, SIZE), size=RED_SIZE, mode='bilinear', align_corners=True)
        #print(gt_obs.size())

        recon_batch, gt_obs_mu, gt_obs_logsigma = self.vae(gt_obs)
        #print(gt_obs_mu.size())
        #print(BSIZE, SEQ_LEN, LSIZE)
        latents = (gt_obs_mu + gt_obs_logsigma.exp() * torch.randn_like(gt_obs_mu)).view(BSIZE, SEQ_LEN, LSIZE)

        latents = latents.transpose(1,0)
        ins = torch.cat([actions, latents], dim=-1)
        outs, _ = self.rnn(ins) # RNN stuff
        gmm_outs = self.gmm_linear(outs) # GMM stuff

        stride = self.gaussians * self.latents

        mus = gmm_outs[:, :, :stride]
        mus = mus.view(seq_len, bs, self.gaussians, self.latents)

        sigmas = gmm_outs[:, :, stride:2 * stride]
        sigmas = sigmas.view(seq_len, bs, self.gaussians, self.latents)
        sigmas = torch.exp(sigmas)

        pi = gmm_outs[:, :, 2 * stride: 2 * stride + self.gaussians]
        pi = pi.view(seq_len, bs, self.gaussians)
        logpi = f.log_softmax(pi, dim=-1)

        rs = gmm_outs[:, :, -2]

        ds = gmm_outs[:, :, -1]

        #Turn mus into observations
        n_gauss = mus.size(2)
        mu_vae = mus.contiguous().view(-1, self.latents)
        vae_obs = self.vae.decoder(mu_vae)
        recon_vae_obs = vae_obs.view(SEQ_LEN, BSIZE, n_gauss, -1)

        # n_gauss = sigmas.size(2)
        # sigma_vae = sigmas.contiguous().view(-1, self.latents)
        # vae_sigma = self.vae.decoder(sigma_vae)
        # sigma_obs = vae_sigma.view(SEQ_LEN, BSIZE, n_gauss, -1)

        return mus, recon_vae_obs, sigmas, logpi, rs, ds, recon_batch, latents

# class VAERNNCell(_VAERNNBase):
#     """ MDRNN model for one step forward """
#     def __init__(self,latents, actions, hiddens, gaussians):
#         super().__init__(latents, actions, hiddens, gaussians)
#         self.rnn = nn.LSTMCell(latents + actions, hiddens)
#         self.vae = VAE(3, latents)
#
#     def forward(self, action, ob, hidden): # pylint: disable=arguments-differ
#         """ ONE STEP forward.
#
#         :args actions: (BSIZE, ASIZE) torch tensor
#         :args latents: (BSIZE, LSIZE) torch tensor
#         :args hidden: (BSIZE, RSIZE) torch tensor
#
#         :returns: mu_nlat, sig_nlat, pi_nlat, r, d, next_hidden, parameters of
#         the GMM prediction for the next latent, gaussian prediction of the
#         reward, logit prediction of terminality and next hidden state.
#             - mu_nlat: (BSIZE, N_GAUSS, LSIZE) torch tensor
#             - sigma_nlat: (BSIZE, N_GAUSS, LSIZE) torch tensor
#             - logpi_nlat: (BSIZE, N_GAUSS) torch tensor
#             - rs: (BSIZE) torch tensor
#             - ds: (BSIZE) torch tensor
#         """
#         recon_batch, latent, logvar = self.vae(ob)
#
#         in_al = torch.cat([action, latent], dim=1)
#
#         next_hidden = self.rnn(in_al, hidden)
#         out_rnn = next_hidden[0]
#
#         out_full = self.gmm_linear(out_rnn)
#
#         stride = self.gaussians * self.latents
#
#         mus = out_full[:, :stride]
#         mus = mus.view(-1, self.gaussians, self.latents)
#
#         sigmas = out_full[:, stride:2 * stride]
#         sigmas = sigmas.view(-1, self.gaussians, self.latents)
#         sigmas = torch.exp(sigmas)
#
#         pi = out_full[:, 2 * stride:2 * stride + self.gaussians]
#         pi = pi.view(-1, self.gaussians)
#         logpi = f.log_softmax(pi, dim=-1)
#
#         r = out_full[:, -2]
#
#         d = out_full[:, -1]
#
#         #Turn mus into observations
#         BSIZE, n_gauss = mus.size(0), mus.size(1)
#         mu_vae = mus.view(-1, self.latents)
#         vae_obs = self.vae.decoder(mu_vae)
#         flat_vae_obs = vae_obs.view(BSIZE, n_gauss, -1)
#
#         return mus, flat_vae_obs, sigmas, logpi, r, d, next_hidden, recon_batch, latent, logvar
