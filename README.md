# Enhanced World Models for Goal-Driven Learning
Code for CS 231N Project by Kent Vainio, Sharman Tan, and Megumi Sano

## Where to find stuff 
1. Vanilla VAE (trained for 50 epochs): "/home/megumisano/world-models/exp_dir/vae"
2. VAE with reward (trained for 50 epochs): "/home/megumisano/world-models/exp_dir/vae_reward_eval"
3. VGG-16 encoder-decoder (trained for 50 epochs): "/home/gengar888/world-models/exp_dir/vgg"
4. Rollouts: "/home/gengar888/world-models/rollouts" 
5. MDRNN trained separately from vanilla VAE (trained for default # epochs): "/home/megumisano/world-models/exp_dir/mdrnn"
6. VAE-RNN (trained for 20 epochs): "/home/gengar888/enhanced_world_models/vaernn_no_gmm"

Still need to...
- [ ] Train MDRNN on VAE with reward 
- [ ] Train controller on MDRNN with vanilla VAE 
- [ ] Train controller on MDRNN with VAE with reward 
- [ ] Train controller on VAE-RNN 
- [ ] Maybe VGG? 

## TODO's 
- [ ] Milestone (due Wed 5/15) 
  - [x] Run WorldModelsExperiments 
  - [x] Implement simple baseline 
  - [ ] Train V and M together 
  - [ ] Train with attention 
  - [ ] Try out a different environment? Atari

- [ ] Final Report (Tues 6/4) 
 - [ ] Abstract 
 - [ ] Introduction
 - [ ] Methods 
  - [ ] Train VAE with reward (include fig and eqs) 
  - [ ] Train VAE and RNN together with RNN predicting reward (include fig and eqs) 
 - [ ] Experiments 
  - [ ] Train controller on baseline 
  - [ ] Train controller on baseline but with VAE with reward 
  - [ ] Train controller on vae-rnn 
  - [ ] Compare reconstructions for all of the above 
 - [ ] Conclusion 
 
- [ ] Poster Session (Tues 6/11) 

** Most of repo content as of now comes from https://github.com/ctallec/world-models **
