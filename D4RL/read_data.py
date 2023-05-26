import h5py
f1 = h5py.File('/home/sarthak/.d4rl/datasets/carla_lane_follow_flat-v0.hdf5', 'r')

# print (f1.keys()) --> <KeysViewHDF5 ['actions', 'observations', 'rewards', 'terminals', 'timeouts']>

actions = f1['actions']
# action: (100000, 2)

observations = f1['observations']
# observations: (100000, 6912)

rewards = f1['rewards']
terminals = f1['terminals']
timeouts = f1['timeouts']
# (100000,) (100000,) (100000,)
