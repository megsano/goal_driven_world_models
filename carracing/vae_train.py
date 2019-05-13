'''
Train VAE model on data created using extract.py
final model saved into tf_vae/vae.json
'''

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" # can just override for multi-gpu systems

import tensorflow as tf
import random
import numpy as np
np.set_printoptions(precision=4, edgeitems=6, linewidth=100, suppress=True)

MODEL = 'resnet50'

if MODEL == 'vae':
  from vae.vae import ConvVAE, reset_graph 
elif MODEL == 'resnet50':
  from vae.vae_resnet import ConvVAE, reset_graph # changed to vae_resnet 

# Hyperparameters for ConvVAE
z_size=32
batch_size=100
learning_rate=0.0001
kl_tolerance=0.5

# Parameters for training
NUM_EPOCH = 10
DATA_DIR = "record"

model_save_path = "tf_vae"
if not os.path.exists(model_save_path):
  os.makedirs(model_save_path)

def count_length_of_filelist(filelist):
  # although this is inefficient, much faster than doing np.concatenate([giant list of blobs])..
  N = len(filelist)
  total_length = 0
  for i in range(N):
    filename = filelist[i]
    raw_data = np.load(os.path.join("record", filename))['obs']
    l = len(raw_data)
    total_length += l
    if (i % 1000 == 0):
      print("loading file", i)
  return  total_length

def create_dataset(filelist, N=160, M=1000): # N is 10000 episodes, M is number of timesteps # 160 for training 
  data = np.zeros((M*N, 64, 64, 3), dtype=np.uint8)
  idx = 0
  for i in range(N):
    filename = filelist[i]
    raw_data = np.load(os.path.join("record", filename))['obs']
    l = len(raw_data)
    if (idx+l) > (M*N):
      data = data[0:idx]
      print('premature break')
      break
    data[idx:idx+l] = raw_data
    idx += l
    if ((i+1) % 100 == 0):
      print("loading file", i+1)
  return data

# load dataset from record/*. only use first 10K, sorted by filename.
filelist = os.listdir(DATA_DIR)
filelist.sort()
filelist = filelist[0:160] # just on training split 
print("check total number of images:", count_length_of_filelist(filelist))
dataset = create_dataset(filelist)

# train_datagen = ImageDataGenerator(
#         preprocessing_function=preprocess_input,
#         rotation_range=90,
#         horizontal_flip=True,
#         vertical_flip=False
#       )

#       TRAIN_DIR = 
#       train_generator = train.datagetn.flow_from_directory(TRAIN_DIR,
#                                                           target_size=(HEIGHT, WIDTH),
#                                                           batch_size=self.batch_size
#                                                           )

# split into batches:
total_length = len(dataset)
num_batches = int(np.floor(total_length/batch_size))
print("num_batches", num_batches)

reset_graph()

vae = ConvVAE(z_size=z_size,
              batch_size=batch_size,
              learning_rate=learning_rate,
              kl_tolerance=kl_tolerance,
              is_training=True,
              reuse=False,
              gpu_mode=True)

# train loop:
print("train", "step", "loss", "recon_loss", "kl_loss")

for epoch in range(NUM_EPOCH):

  print("epoch : {}".format(epoch))

  np.random.shuffle(dataset)
  for idx in range(num_batches):
    batch = dataset[idx*batch_size:(idx+1)*batch_size]

    obs = batch.astype(np.float)/255.0

    feed = {vae.x: obs,}
    print("obs.shape: {}".format(obs.shape))

    if MODEL == 'vae':
      (train_loss, r_loss, kl_loss, train_step, _) = vae.sess.run([
        vae.loss, vae.r_loss, vae.kl_loss, vae.global_step, vae.train_op
      ], feed)
      if ((train_step+1) % 500 == 0):
        print("step", (train_step+1), train_loss, r_loss, kl_loss)
      if ((train_step+1) % 5000 == 0):
        vae.save_json("tf_vae/vae_new.json")

    elif MODEL == 'resnet50':
      (train_loss, r_loss, train_step, _) = vae.sess.run([
        vae.loss, vae.r_loss, vae.global_step, vae.train_op
      ], feed)
      if ((train_step+1) % 500 == 0):
        print("step", (train_step+1), train_loss, r_loss)
      if ((train_step+1) % 5000 == 0):
        vae.save_json("tf_vae/vae_resnet.json")
    

# finished, final model:
if MODEL == 'vae':
  vae.save_json("tf_vae/vae_new.json")
elif MODEL == 'resnet50':
  vae.save_json("tf_vae/vae_resnet.json")
