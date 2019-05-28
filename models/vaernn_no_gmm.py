"""
Define the VAE RNN with NO GMM.
"""
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.distributions.normal import Normal
from models.vae import VAE
from utils.misc import ASIZE, LSIZE, RSIZE, RED_SIZE, SIZE

def obs_loss(gt_next_obs, recon_next_obs):
    gt_next_obs = gt_next_obs.transpose(1,0)
    gt_next_obs = f.upsample(gt_next_obs.view(-1, 3, SIZE, SIZE), size=RED_SIZE, mode='bilinear', align_corners=True)
    
    return f.mse_loss(gt_next_obs, recon_next_obs, size_average=False)

class _VAERNN_NOGMMBase(nn.Module):
    def __init__(self, latents, actions, hiddens, gaussians):
        super().__init__()
        self.latents = latents
        self.actions = actions
        self.hiddens = hiddens
        self.obs_linear = nn.Linear(hiddens, 2 * latents + 2)

    def forward(self, *inputs):
        pass

class VAERNN_NOGMM(_VAERNN_NOGMMBase):
    """ VAERNN NO GMM model for multi steps forward """
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

        gt_obs = gt_obs.transpose(1,0)
        gt_obs = f.upsample(gt_obs.view(-1, 3, SIZE, SIZE), size=RED_SIZE, mode='bilinear', align_corners=True)

        recon_batch, gt_obs_mu, gt_obs_logsigma = self.vae(gt_obs)
        latents = (gt_obs_mu + gt_obs_logsigma.exp() * torch.randn_like(gt_obs_mu)).view(BSIZE, SEQ_LEN, LSIZE)

        latents = latents.transpose(1,0)
        ins = torch.cat([actions, latents], dim=-1)
        outs, _ = self.rnn(ins) # RNN stuff
        latent_outs = self.obs_linear(outs) # GMM stuff
            
        mus = latent_outs[:, :, :self.latents]
        mus = mus.view(SEQ_LEN * BSIZE, -1)
        sigmas = latent_outs[:, :, self.latents:self.latents*2]
        sigmas = sigmas.view(SEQ_LEN * BSIZE, -1)
        rs = latent_outs[:, :, -2]
        ds = latent_outs[:, :, -1]
        
        ls = (mus + sigmas.exp() * torch.randn_like(mus)).view(SEQ_LEN * BSIZE, -1)
        
        vae_obs = self.vae.decoder(ls)
        recon_vae_obs = vae_obs.view(SEQ_LEN * BSIZE, 3, 64, 64)

        return recon_vae_obs, ls, rs, ds, recon_batch, gt_obs_mu, gt_obs_logsigma, mus, sigmas
 
 #  class VAERNN_NOGMMCell(_VAERNN_NOGMMBase):
#     """ VAERNN NO GMM Cell model for one step forward """
#     def __init__(self,latents, actions, hiddens, gaussians):
#         super().__init__(latents, actions, hiddens, gaussians)
#         self.rnn = nn.LSTMCell(latents + actions, hiddens)
#         self.vae = VAE(3, latents)

#     def forward(self, action, gt_ob, hidden):
#         """ ONE STEP forward.

#         :args actions: (BSIZE, ASIZE) torch tensor
#         :args latents: (BSIZE, LSIZE) torch tensor
#         :args hidden: (BSIZE, RSIZE) torch tensor

#         :returns: mu_nlat, sig_nlat, pi_nlat, r, d, next_hidden, parameters of
#         the GMM prediction for the next latent, gaussian prediction of the
#         reward, logit prediction of terminality and next hidden state.
#             - mu_nlat: (BSIZE, N_GAUSS, LSIZE) torch tensor
#             - sigma_nlat: (BSIZE, N_GAUSS, LSIZE) torch tensor
#             - logpi_nlat: (BSIZE, N_GAUSS) torch tensor
#             - rs: (BSIZE) torch tensor
#             - ds: (BSIZE) torch tensor
#         """
#         SEQ_LEN, BSIZE = actions.size(0), actions.size(1) # batch of actions for each time step in the sequence (t steps) T x N  x A
#         bs, seq_len = BSIZE, SEQ_LEN

#         gt_obs = gt_obs.transpose(1,0)
#         gt_obs = f.upsample(gt_obs.view(-1, 3, SIZE, SIZE), size=RED_SIZE, mode='bilinear', align_corners=True)

#         recon_batch, gt_obs_mu, gt_obs_logsigma = self.vae(gt_obs)
#         latents = (gt_obs_mu + gt_obs_logsigma.exp() * torch.randn_like(gt_obs_mu)).view(BSIZE, SEQ_LEN, LSIZE)

#         latents = latents.transpose(1,0)
#         ins = torch.cat([actions, latents], dim=-1)
#         outs, _ = self.rnn(ins) # RNN stuff
#         latent_outs = self.obs_linear(outs) # GMM stuff
            
#         mus = latent_outs[:, :, :self.latents]
#         mus = mus.view(SEQ_LEN * BSIZE, -1)
#         sigmas = latent_outs[:, :, self.latents:self.latents*2]
#         sigmas = sigmas.view(SEQ_LEN * BSIZE, -1)
#         rs = latent_outs[:, :, -2]
#         ds = latent_outs[:, :, -1]
        
#         ls = (mus + sigmas.exp() * torch.randn_like(mus)).view(SEQ_LEN * BSIZE, -1)
        
#         vae_obs = self.vae.decoder(ls)
#         recon_vae_obs = vae_obs.view(SEQ_LEN * BSIZE, 3, 64, 64)

#         return recon_vae_obs, latents, rs, ds, recon_batch, gt_obs_mu, gt_obs_logsigma, mus, sigmas
        
        
#         #######
        
#         recon_batch, latent, logvar = self.vae(ob)

#         in_al = torch.cat([action, latent], dim=1)

#         next_hidden = self.rnn(in_al, hidden)
#         out_rnn = next_hidden[0]

#         out_full = self.gmm_linear(out_rnn)

#         stride = self.gaussians * self.latents

#         mus = out_full[:, :stride]
#         mus = mus.view(-1, self.gaussians, self.latents)

#         sigmas = out_full[:, stride:2 * stride]
#         sigmas = sigmas.view(-1, self.gaussians, self.latents)
#         sigmas = torch.exp(sigmas)

#         pi = out_full[:, 2 * stride:2 * stride + self.gaussians]
#         pi = pi.view(-1, self.gaussians)
#         logpi = f.log_softmax(pi, dim=-1)

#         r = out_full[:, -2]

#         d = out_full[:, -1]

#         #Turn mus into observations
#         BSIZE, n_gauss = mus.size(0), mus.size(1)
#         mu_vae = mus.view(-1, self.latents)
#         vae_obs = self.vae.decoder(mu_vae)
#         flat_vae_obs = vae_obs.view(BSIZE, n_gauss, -1)

#         return mus, flat_vae_obs, sigmas, logpi, r, d, next_hidden, recon_batch, latent, logvar
