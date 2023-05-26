import h5py
import numpy as np
import torch
from matplotlib.image import imsave

# f1 = h5py.File('/home/sarthak/.d4rl/datasets/carla_lane_follow_flat-v0.hdf5', 'r')
f1 = h5py.File('carla_waypoints.hdf5', 'r')

# print (f1.keys()) --> <KeysViewHDF5 ['actions', 'observations', 'rewards', 'terminals', 'timeouts']>

actions = f1['actions']
# action: (100000, 2)

observations = f1['observations']
# observations: (100000, 6912)

rewards = f1['rewards']
terminals = f1['terminals']
timeouts = f1['timeouts']
# (100000,) (100000,) (100000,)

start = 0
save_num = 200

for i in range(start, start+save_num):
    obs = observations[i]
    obs = obs.reshape(48, 48, 3)
    imsave('./saved_imgs/{}.jpg'.format(i), obs)


