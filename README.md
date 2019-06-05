# Goal-Driven World Models
Code for CS 231N Project by Kent Vainio, Sharman Tan, and Megumi Sano

## To train world models: 
1. run `python trainmdrnn.py` with vae model either set to `vae` or `vae_reward_eval` (must run `python trainvae.py` or `python trainvaereward.py` to train VAE first) 
2. run `python trainvaernn.py` to train VAE-RNN-G 
3. run `python trainvaernn_no_gmm.py` to train VAE-RNN-O 

## To train controller:
4. run `python traincontroller.py` with desired world model folder below: 

## Model weights 
1. Vanilla VAE (trained for 50 epochs): "/home/megumisano/world-models/exp_dir/vae"
2. VAE with reward (trained for 50 epochs): "/home/megumisano/world-models/exp_dir/vae_reward_eval"
3. VGG-16 encoder-decoder (trained for 50 epochs): "/home/gengar888/world-models/exp_dir/vgg"
4. Rollouts: "/home/gengar888/world-models/rollouts" 
5. MDRNN trained separately from vanilla VAE (trained for default # epochs): "/home/megumisano/world-models/exp_dir/mdrnn"
6. MDRNN trained separately from VAE with reward (trained for default # epochs): "/home/megumisano/world-models/exp_dir/mdrnn_vae_reward"
7. VAE-RNN (trained for 50 epochs): "/home/gengar888/enhanced_world_models/vaernn_no_gmm"
8. Controller trained on MDRNN (trained for 100 iterations with pop-size 16): "/home/megumisano/world-models/exp_dir/ctrl"
9. Controller trained on VAERNN (trained for 100 iterations with pop-size 16): "/home/megumisano/world-models/exp_dir/ctrl_vaernn"
10. Controller trained on mdrnn_vae_reward (trained for 100 iterations with pop-size 16): "/home/megumisano/world-models/exp_dir/ctrl_vae_reward"

** Most of repo content as of now comes from https://github.com/ctallec/world-models **
