"""
Utility functions as well as training code for the distributions/models on the FetchSweep environment. 
Requires the source dataset to have been collected, which can be done via mrl codebase. 

Example command:

  python augment_offline_fetch.py

"smaller state" data has format:

obseration:
0-2: grip pos
3-5: grip velp
6-8: o1 pos
9-10: o1 xyvel
11-13: o2 pos
14-15: o2 xyvel

desired goal:
0-2: o1 target
3-5: o2 target


"""

from hashlib import sha1
from augment_offline import *
from density_utils import *

import multiprocessing as mp
import os
import pickle
import sys
from copy import deepcopy

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from itertools import chain

# To define a causal model, we first define the slots, the slot-lengths, and the parents
FETCH_SLOTS_SMALLER = [
  ('grip_1', 6, []),
  ('ob1_1', 5, []),
  ('ob2_1', 5, []),
  #('dg_ob1_1', 3, []),
  #('dg_ob2_1', 3, []),
  ('act_pos', 4, []),
  ('grip_2', 6, ['grip_1', 'act_pos']),
  ('ob1_2', 5, ['grip_1', 'ob1_1', 'act_pos']),
  ('ob2_2', 5, ['grip_1', 'ob2_1', 'act_pos']),
  #('dg_ob1_2', 3, ['dg_ob1_1']),
  #('dg_ob2_2', 3, ['dg_ob2_1']),
]

FETCH_SLOTS = [
  ('grip_1', 10, []),
  ('ob1_1', 12, []),
  ('ob2_1', 12, []),
  #('dg_ob1_1', 3, []),
  #('dg_ob2_1', 3, []),
  ('act_pos', 4, []),
  ('grip_2', 10, ['grip_1', 'act_pos']),
  ('ob1_2', 12, ['grip_1', 'ob1_1', 'act_pos']),
  ('ob2_2', 12, ['grip_1', 'ob2_1', 'act_pos']),
  #('dg_ob1_2', 3, ['dg_ob1_1']),
  #('dg_ob2_2', 3, ['dg_ob2_1']),
]


FETCH_RAW_SLOTS = [
  ('time_1', 1, []),
  ('grip_1', 13, []),
  ('ob1_1', 7, []),
  ('ob2_1', 7, []),
  ('grip_1v', 13, []),
  ('ob1_1v', 6, []),
  ('ob2_1v', 6, []), # end of obs

  ('dg_ob1_1', 3, []),
  ('dg_ob2_1', 3, []),

  ('act_pos', 4, []),

  ('time_2', 1, ['time_1']),
  ('grip_2', 13, ['grip_1', 'grip_1v', 'act_pos']),
  ('ob1_2', 7, ['grip_1', 'ob1_1', 'grip_1v', 'ob1_1v', 'act_pos']),
  ('ob2_2', 7, ['grip_1', 'ob2_1', 'grip_1v', 'ob2_1v', 'act_pos']),
  ('grip_2v', 13, ['grip_1', 'grip_1v', 'act_pos']),
  ('ob1_2v', 6, ['grip_1', 'ob1_1', 'grip_1v', 'ob1_1v', 'act_pos']),
  ('ob2_2v', 6, ['grip_1', 'ob2_1', 'grip_1v', 'ob2_1v', 'act_pos']), # end of obs

  ('dg_ob1_2', 3, ['dg_ob1_1']),
  ('dg_ob2_2', 3, ['dg_ob2_1']),
]

def format_nodes(nodes):
  res = []
  idx = 0
  for node in nodes:
    node_name, node_length, parents = node
    res.append((node_name, {'idxs': list(range(idx, idx+node_length)), 'parents': parents}))
    idx += node_length
  return res

FETCH_NODES_SMALLER = format_nodes(FETCH_SLOTS_SMALLER)
FETCH_NODES = format_nodes(FETCH_SLOTS)
FETCH_RAW_NODES = format_nodes(FETCH_RAW_SLOTS)

def import_env(parent_folder, place_two=True, smaller_state=True, raw_state=False):
  sys.path.append(parent_folder)
  from envs.customfetch.custom_fetch import FetchHookSweepAllEnv
  if raw_state:
    return FetchHookSweepAllEnv(place_two=place_two, raw_state=True)
  return FetchHookSweepAllEnv(place_two=place_two, smaller_state=smaller_state)

def import_dataset(filepath):
  with open(filepath, 'rb') as f:
    buffer = pickle.load(f)
  return buffer

def process_fetchtransition(t, use_random_dg=False, obs_size=16):
  """ 
  Takes a fetch transition in original format, and returns the (s, a, s') with dg concatenated.
  """
  s1, a, s2, r, d = t
  _s1 = np.zeros((obs_size + 6,))
  _s2 = np.zeros((obs_size + 6,))
  if use_random_dg:
    dg = ENV.reset()['desired_goal']
  else:
    dg = s1['desired_goal']

  _s1[-6:] = dg
  _s2[-6:] = dg
  _s1[:-6] = s1['observation']
  _s2[:-6] = s2['observation']
  return (_s1, a[0], _s2)

def process_raw_fetchtransition(t):
  return process_fetchtransition(t, obs_size=53)

def process_buffer(buffer, n_workers=8):
  data = []
  splits = np.array_split(np.array(buffer, dtype='object'), 50)
  with mp.Pool(n_workers, initializer = lambda: np.random.seed() ) as pool:
    for i in tqdm.tqdm(range(len(splits))):
      data += pool.map(process_fetchtransition, splits[i])
      
  print(f"Processed {len(data)} transitions")
  return data

def process_raw_buffer(buffer, n_workers=8):
  data = []
  splits = np.array_split(np.array(buffer, dtype='object'), 50)
  with mp.Pool(n_workers, initializer = lambda: np.random.seed() ) as pool:
    for i in tqdm.tqdm(range(len(splits))):
      data += pool.map(process_raw_fetchtransition, splits[i])
      
  print(f"Processed {len(data)} transitions")
  return data

def sasrd_to_fetch_transition(t):
  """ 
  Takes a single fetch transition in sasrd (output of the augmentation) format, and returns the origina original format.
  """
  return [
    {'observation': t[:16],
     'achieved_goal': t[[6,7,8,11,12,13]],
     'desired_goal': t[16:22]},
    t[22:26],
    {'observation': t[26:42],
     'achieved_goal': t[[32,33,34,37,38,39]],
     'desired_goal': t[42:48]},
    t[48],
    t[49]
  ]

def unprocess_buffer(sasrd, n_workers=8):
  data = []
  splits = np.array_split(np.array(sasrd, dtype='float16'), 50)
  with mp.Pool(n_workers, initializer = lambda: np.random.seed() ) as pool:
    for i in tqdm.tqdm(range(len(splits))):
      data += pool.map(sasrd_to_fetch_transition, splits[i])
      
  print(f"Processed {len(data)} transitions")
  return data


def plot_sample(sample, normalizer=None, num_p=500, smaller=True):
  plt.figure()
  if normalizer:
    sample = normalizer.inverse_transform(sample)

  O1X = 6 if smaller else 10
  O2X = 11 if smaller else 22

  plt.rcParams["figure.figsize"] = (12,8)
  plt.scatter(sample[:num_p,O1X], sample[:num_p,O1X+1], s=11, c='black', alpha=0.4)
  plt.scatter(sample[:num_p,O2X], sample[:num_p,O2X+1], s=11, c='red', alpha=0.4)
  for s in sample[:num_p]:
    plt.plot([s[O1X],s[O2X]], [s[O1X+1],s[O2X+1]], c='b', alpha=0.15)

  plt.ylim(0.5, 1.0)
  plt.xlim(1.3, 2.4)

def plot_sample2(sample, normalizer=None, num_p=10000, smaller=True, filename=None, errs=None, ax=None):

  if ax is None:
    plt.rcParams["figure.figsize"] = (12, 8)
    fig, ax = plt.subplots()

  if normalizer:
    sample = normalizer.inverse_transform(sample)

  O1X = 6 if smaller else 10
  O2X = 11 if smaller else 22
  
  if errs is not None:
    errs = errs[np.linalg.norm(sample[:,O1X:O1X+2] - sample[:,O2X:O2X+2], axis=-1) < 0.2]
    errs = errs[:num_p]

  sample = sample[np.linalg.norm(sample[:,O1X:O1X+2] - sample[:,O2X:O2X+2], axis=-1) < 0.2]
  sample = sample[:num_p]

  #plt.rcParams["contour.linewidth"] = 0
  #gx = np.concatenate((sample[:,16], sample[:,19]))
  #gy = np.concatenate((sample[:,17], sample[:,20]))
  #gy = gy[np.logical_or(gx <1.7, gx > 2.0)]
  #gx = gx[np.logical_or(gx <1.7, gx > 2.0)]

  #sns.kdeplot(gx, y=gy, shade=True, levels=40, color='green', antialiased=True, alpha=0.5, linewidth=0, thresh=0.2, bw=0.02),
  #plt.scatter(gx, gy, s=7, c='green', alpha=0.2)

  if errs is None:
    ax.scatter((sample[:,O1X] + sample[:,O2X]) / 2., (sample[:,O1X+1] + sample[:,O2X+1]) / 2., s=11, c='b', alpha=0.2)
    sns.kdeplot((sample[:,O1X] + sample[:,O2X]) / 2., y=(sample[:,O1X+1] + sample[:,O2X+1]) / 2, color='blue', shade=True, levels=11, antialiased=True, alpha=0.4, thresh=0.02, bw_method=0.3, ax=ax)
  else:
    max_err = 0.04
    midpoints = np.stack(((sample[:,O1X] + sample[:,O2X]) / 2., (sample[:,O1X+1] + sample[:,O2X+1]) / 2.), axis=1)
    for s,e  in zip(midpoints,errs):
      ax.add_patch(plt.Circle(s, 0.005, color='r', alpha=min(e/max_err, 1.)))
    #ax.scatter((sample[:,O1X] + sample[:,O2X]) / 2., (sample[:,O1X+1] + sample[:,O2X+1]) / 2., s=11, c='b', alpha=min(e/max_err, 1.))

  ax.set_ylim(0.5, 1.0)
  ax.set_xlim(1.3, 2.4)

  if filename is not None:
    plt.savefig(filename)

def plot_sample3(sample, normalizer=None, num_p=10000, filename=None):
  plt.figure()
  if normalizer:
    sample = normalizer.inverse_transform(sample)
  
  sample = sample[np.linalg.norm(sample[:,14:16] - sample[:,21:23], axis=-1) < 0.2]
  sample = sample[:num_p]

  plt.rcParams["figure.figsize"] = (12,8)
  plt.rcParams["contour.linewidth"] = 0
  gx = np.concatenate((sample[:,53], sample[:,56]))
  gy = np.concatenate((sample[:,54], sample[:,57]))
  gy = gy[np.logical_or(gx <1.7, gx > 2.0)]
  gx = gx[np.logical_or(gx <1.7, gx > 2.0)]

  sns.kdeplot(gx, y=gy, shade=True, levels=40, color='green', antialiased=True, alpha=0.15, thresh=0.02, bw_method=0.05),
  plt.scatter(gx, gy, s=7, c='green', alpha=0.2)

  plt.scatter((sample[:,14] + sample[:,21]) / 2., (sample[:,15] + sample[:,22]) / 2., s=11, c='b', alpha=0.2)
  sns.kdeplot((sample[:,14] + sample[:,21]) / 2., y=(sample[:,15] + sample[:,22]) / 2, color='blue', shade=True, 
      levels=40, antialiased=True, alpha=0.6, thresh=0.005, bw_method=0.3)

  plt.ylim(0.5, 1.0)
  plt.xlim(1.3, 2.4)

  if filename is not None:
    plt.savefig(filename)

def maybe_np(tensor_or_array):
  try:
    return tensor_or_array.cpu().numpy()
  except:
    return tensor_or_array

def normalize_and_prune(dataset, stds = 4.):
  normalizer = StandardScaler()
  normalized_data = normalizer.fit_transform(dataset)
  normalized_data = normalized_data[(np.abs(normalized_data) < stds).all(axis=1)]
  normalized_data = normalized_data[np.random.permutation(range(len(normalized_data)))]
  print(f'Pruned dataset has length {len(normalized_data)}')
  return normalizer, normalized_data

def compare_models_on_dist(models, dist, dist_name="", axes=None):
  if axes is None:
    plt.rcParams["figure.figsize"] = (1 + 3 * len(models), 3)
    fig, axes = plt.subplots(1, len(models))

  print(dist_name)
  res = []
  for (name, model), ax in zip(models, axes):
    errs = np.linalg.norm((model(dist) - fwd(dist))[:,-2:], axis=1)
    print(f"model: {name:4}, error: {errs.mean():.5f}")
    res.append(errs.mean())
    plot_sample2(model(dist), num_p=500, errs=errs, ax=ax)

  return res

def compare_models_on_dists(models, dists, folder=None, seed=""):
  plt.rcParams["figure.figsize"] = (1 + 3 * len(models), 3 * len(dists))
  fig, axes = plt.subplots(len(dists), len(models))
  
  res = {'models': [model[0] for model in models]}
  for dist, axx in zip(dists, axes):
    errs = compare_models_on_dist(models, dist[0], dist[1], axx)
    res[dist[1]] = errs

  if folder:
    with open(os.path.join(folder, f'compare_models_seed{seed}.pickle'), 'wb') as f:
      pickle.dump(res, f)
    plt.savefig(os.path.join(folder, f'compare_models_seed{seed}.png'))

def reward_and_done_for_fetch(sas, use_target_goals, use_her=False, env=None, smaller_state=True, bs=10000):
  dgs = []
  rs = []
  ds = []
  for batch in np.array_split(sas, len(sas) // bs):
    
    if use_target_goals:
      dg = np.stack([env._sample_goal() for i in range(len(batch))])
    else:
      raise NotImplementedError
      dg = batch[:, [42, 43, 44, 45, 46, 47]]

    if smaller_state:
      r = env.compute_reward(batch[:, [26, 27, 28, 31, 32, 33]], 
                       dg, 
                       {'s': batch[:,[20, 21, 22]]})
      if use_her:
        i = np.logical_and( r < -0.3 , np.random.uniform(size=r.shape) < 0.1)[:,None]
        i = np.where(i)[0]
        ag = batch[:, [26, 27, 28, 31, 32, 33]]
        dg[i] = ag[i]

        r = env.compute_reward(batch[:, [26, 27, 28, 31, 32, 33]], 
                        dg, 
                        {'s': batch[:,[20, 21, 22]]})
      dgs.append(dg[:,[0,1,3,4]])


    else:
      raise
      r = env.compute_reward(batch[:, [48, 49, 50, 60, 61, 62]], 
                       dg, 
                       {'s': batch[:,[38, 39, 40]]})


    d = r > -0.3
    
    rs.append(r)
    ds.append(d)
    
  return np.concatenate(dgs), np.concatenate(rs), np.concatenate(ds)

def prune_to_uniform(proposals, target_size=12000., smaller=True):
  from sklearn.neighbors import KernelDensity
  sample = proposals[-10000:]
  O1X = 6 if smaller else 10
  O2X = 11 if smaller else 22

  fmap = lambda s: s[:,[O1X,O1X+1,O2X,O2X+1]]
  K = KernelDensity(bandwidth=0.05)
  K.fit(fmap(sample))
  scores = K.score_samples(fmap(proposals))
  scores = np.maximum(scores, np.log(0.01))
  scores = (1. / np.exp(scores))
  scores = scores / scores.mean()  * (target_size / len(proposals))
  
  return proposals[np.random.uniform(size=scores.shape) < scores]   

def prune_to_uniform2(proposals, target_size=12000., smaller=True):
  from sklearn.neighbors import KernelDensity
  O1X = 6 if smaller else 10
  O2X = 11 if smaller else 22

  proposals = proposals[np.linalg.norm(proposals[:,O1X:O1X+2] - proposals[:,O2X:O2X+2], axis=-1) < 0.3]
  sample = proposals[-5000:]

  fmap = lambda s: s[:,[O1X,O1X+1,O2X,O2X+1]]
  K = KernelDensity(bandwidth=0.05)
  K.fit(fmap(sample))
  scores = K.score_samples(fmap(proposals))
  scores = np.maximum(scores, np.log(0.05))
  scores = (1. / np.exp(scores))
  if np.minimum(scores, 1).sum() > 10000:
    while np.minimum(scores, 1).sum() > 10000:
      scores = scores * 0.99
  else:
    while np.minimum(scores, 1).sum() < 10000:
      scores = scores / 0.99
  
  return proposals[np.random.uniform(size=scores.shape) < scores]   


def generate_distributions(dataset, parent_set_idxs, smaller=True):
  full_data = dataset[np.random.permutation(range(len(dataset)))]
  tr = full_data[:-10000]
  te = full_data[-10000:]

  M = ConditionalGMMParentsModel(n_components=32)
  with torch.no_grad():
    M.setup_parent_models(tr, parent_set_idxs)
    M.fit_parent_data(tr)

  coda_global = M.generate_parents(10000)

  coda_global_uniform_pruned = prune_to_uniform(M.generate_parents(100000), 12000., smaller=smaller)

  return M, [
    tr,
    te,
    coda_global,
    coda_global_uniform_pruned
  ]

def plot_dists(DISTS, saveas=None):
  plt.rcParams["figure.figsize"] = (16,12)
  fig, axes = plt.subplots(2, 3)

  for ax, d in zip(chain.from_iterable(axes), DISTS):
    plot_sample2(d[:10000], ax=ax)
  
  if saveas:
    fig.savefig(saveas)

def plot_training_curves(tr, val, saveas=None):
  plt.rcParams["figure.figsize"] = (4, 3)
  plt.figure()
  plt.plot(tr)
  plt.plot(val)

  if saveas:
    plt.savefig(saveas)


def generate_dummy_rc(roots,children):
  r, c = set(), set()
  for child in children.values():
    for p in child['parents']:
      r.update(roots[p]['idxs'])
    c.update(child['idxs'])
  return {'r': {'idxs': list(r), 'parents': []}},\
        {'c': {'idxs': list(c), 'parents': ['r']}}



def train_fully_connected(base_model, roots, children, training_data, num_epochs=400, lr=1e-4, folder=None):

  model = MLPDynamicsModel(deepcopy(base_model), hidden_layers=[512,512], lr=lr)
  r_all, c_all = generate_dummy_rc(roots, children)
  model.setup_dynamics_models(training_data, r_all, c_all)

  tr_FC, val_FC = model.fit_dynamics(training_data, num_epochs)
  plot_training_curves(tr_FC, val_FC, 
    os.path.join(folder, 'learning_curve_fc.png') if folder else None)

  fwd_fc = lambda X: np.concatenate((X, model.generate_children(X)), 1)

  return model, fwd_fc

def train_disentangled(base_model, roots, children, training_data, num_epochs=400, lr=1e-4, folder=None):
  model = MLPDynamicsModel(deepcopy(base_model), hidden_layers=[512,512], lr=lr)
  model.setup_dynamics_models(training_data, roots, children)
  tr_DS, val_DS = model.fit_dynamics(training_data, num_epochs)
  plot_training_curves(tr_DS, val_DS, 
    os.path.join(folder, 'learning_curve_ds.png') if folder else None)
  fwd_ds = lambda X: np.concatenate((X, model.generate_children(X)), 1)

  return model, fwd_ds

def train_fc_masked(base_model, roots, children, training_data, num_epochs=400, lr=1e-4, folder=None):
  # emulate fully connected w/ Masked Network
  r_all, c_all = generate_dummy_rc(roots, children)
  model = MaskedMLPDynamicsModel(deepcopy(base_model), hidden_layers=[512,512], lr=lr)
  def fc_mask_fn(input_tensor):
    """
    accepts B x total_in_dim and produces B x num_roots x num_children
    returns all 1
    """
    return torch.ones(input_tensor.shape[0], 1, 1, device=input_tensor.device)

  model.setup_dynamics_models(training_data, r_all, c_all, fc_mask_fn)

  tr_MA, val_MA = model.fit_dynamics(training_data, num_epochs)
  plot_training_curves(tr_MA, val_MA, 
    os.path.join(folder, 'learning_curve_mafc.png') if folder else None)


  fwd_ma = lambda X: np.concatenate((X, model.generate_children(X)), 1)

  return model, fwd_ma

def train_ds_masked(base_model, roots, children, training_data, num_epochs=400, lr=1e-4, folder=None):
  # emulate fully connected w/ Masked Network
  model = MaskedMLPDynamicsModel(deepcopy(base_model), hidden_layers=[512,512], lr=lr)
  def fc_mask_fn(input_tensor):
    """
    accepts B x total_in_dim and produces B x num_roots x num_children
    returns all 1
    """
    res = torch.tensor(
      [[1, 1, 1],
      [1, 1, 0],
      [1, 0, 1],
      [1, 1, 1]]
      ).to(input_tensor.device)
    
    return res[None].repeat((input_tensor.shape[0], 1, 1))

  model.setup_dynamics_models(training_data, roots, children, fc_mask_fn)

  tr_MA, val_MA = model.fit_dynamics(training_data, num_epochs)
  plot_training_curves(tr_MA, val_MA, 
    os.path.join(folder, 'learning_curve_mads.png') if folder else None)

  fwd_ma = lambda X: np.concatenate((X, model.generate_children(X)), 1)

  return model, fwd_ma

def train_fetch_masked(base_model, roots, children, training_data, smaller=True, num_epochs=400, lr=1e-4, folder=None):
  # emulate fully connected w/ Masked Network
  model = MaskedMLPDynamicsModel(deepcopy(base_model), hidden_layers=[512,512], lr=lr)
  O1X = 6 if smaller else 10
  O2X = 11 if smaller else 22
  
  def fc_mask_fn(input_tensor):
    """
    accepts B x total_in_dim and produces B x num_roots x num_children
    returns all 1
    """
    res = torch.tensor(
      [[1, 1, 1],
      [1, 1, 0],
      [1, 0, 1],
      [1, 1, 1]]
      ).to(input_tensor.device)
    res = res[None].repeat((input_tensor.shape[0], 1, 1))

    res[torch.sum(torch.abs(input_tensor[:,O1X:O1X+2] - input_tensor[:,O2X:O2X+2]), axis=1) < 0.05] = 1
    
    return res

  model.setup_dynamics_models(training_data, roots, children, fc_mask_fn)

  tr_MA, val_MA = model.fit_dynamics(training_data, num_epochs)
  plot_training_curves(tr_MA, val_MA, 
    os.path.join(folder, 'learning_curve_ma.png') if folder else None)

  fwd_ma = lambda X: np.concatenate((X, model.generate_children(X)), 1)

  return model, fwd_ma

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description='Codav2 on Fetch Sweep')

  parser.add_argument("--env_parent_folder", type=str) # where environment is a module (mrl directory)
  parser.add_argument("--source_folder", type=str) # source path for the og expert dataset


  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--output_folder', type=str, default='./fetch_output')

  parser.add_argument("--smallerstate", type=bool, default=True) 
  parser.add_argument("--n_workers", type=int, default=8) # number of workers to make dataset

  args = parser.parse_args()

  import random
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)

  SMALL_STATE = args.smallerstate
  PARENT_FOLDER = args.env_parent_folder
  SOURCE_FOLDER = args.source_folder
  OUTPUT_FOLDER = args.output_folder
  N_WORKERS = args.n_workers

  # Make output folder if it doesn't exist
  if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

  # Load  & Shuffle the data
  data_place1 = import_dataset(os.path.join(SOURCE_FOLDER, f'noisy_expert_buffer_place1{"small" if SMALL_STATE else ""}.pickle'))
  data_place1 = flatten_dataset([(t[0],t[1][0],t[2]) for t in data_place1])
  data_place1 = data_place1[np.random.permutation(len(data_place1))]

  # data_place2 = import_dataset(os.path.join(SOURCE_FOLDER, f'noisy_expert_buffer_place2{"small" if SMALL_STATE else ""}.pickle'))
  # data_place2 = flatten_dataset([(t[0],t[1][0],t[2]) for t in data_place2])
  # data_place2 = data_place2[np.random.permutation(len(data_place2))]

  # data_placer = import_dataset(os.path.join(SOURCE_FOLDER, f'noisy_expert_buffer_placerandom{"small" if SMALL_STATE else ""}.pickle'))
  # data_placer = flatten_dataset([(t[0],t[1][0],t[2]) for t in data_placer])
  # data_placer = data_placer[np.random.permutation(len(data_placer))]

  flat_sas_data = data_place1

  # Construct Causal Graph and save SCM diagram
  G = make_graph(FETCH_NODES_SMALLER if SMALL_STATE else FETCH_NODES)
  draw_graph(G, os.path.join(OUTPUT_FOLDER, 'global_causal_graph.png'))

  roots, children = split_nodes(G)
  parent_sets = [child['parents'] for child in children.values()]
  parent_set_idxs = []
  for parent_set in parent_sets:
    parent_set_idxs.append(np.concatenate([roots[p]['idxs'] for p in parent_set]))

  #prune_to = 20 if SMALL_STATE else 38
  #normalizer, parent_data = normalize_and_prune(flat_sas_data[:,:prune_to])
  
  # Generate and plot the distributions
  gmm_parents_model, DISTS = generate_distributions(flat_sas_data, parent_set_idxs, smaller=SMALL_STATE)
  tr, te, coda_global, coda_unif = DISTS
  named_dists = list(zip(DISTS, ["tr", "te", "coda_global", "coda_unif"]))

  plot_dists(DISTS, saveas=os.path.join(OUTPUT_FOLDER, 'distributions.png'))

  #  Train all of the models on the training distribution
  #C_FC, fwd_fc = train_fully_connected(gmm_parents_model, roots, children, tr, 4000, lr=2e-4, folder=OUTPUT_FOLDER)
  #C_DS, fwd_ds = train_disentangled(gmm_parents_model, roots, children, tr, 4000, lr=2e-4, folder=OUTPUT_FOLDER)
  #C_MAFC, fwd_mafc = train_fc_masked(gmm_parents_model, roots, children, tr, 4000, lr=2e-4, folder=OUTPUT_FOLDER)
  #C_MADS, fwd_mads = train_ds_masked(gmm_parents_model, roots, children, tr, 4000, lr=2e-4, folder=OUTPUT_FOLDER)
  C_MA, fwd_ma = train_fetch_masked(gmm_parents_model, roots, children, tr, num_epochs=4000, lr=2e-4, folder=OUTPUT_FOLDER)

  models = [('ma', fwd_ma)] #[('fc', fwd_fc), ('ds', fwd_ds), ('mafc', fwd_mafc), ('mads', fwd_mads), ('ma', fwd_ma)]

  # Compare the models across all distributions
  #compare_models_on_dists(models, named_dists, OUTPUT_FOLDER, args.seed)

  # Make the base dataset for downstream task
  env = import_env(PARENT_FOLDER, place_two=True, smaller_state=SMALL_STATE)

  sas = np.concatenate([flat_sas_data]*10) # 5M samples
  dgs, r, d = reward_and_done_for_fetch(sas, use_target_goals=True, env=env, smaller_state=SMALL_STATE)
  if SMALL_STATE:
    s = np.concatenate((sas[:, :16], dgs), axis=1)
    a = sas[:, 16:20]
    ns = np.concatenate((sas[:, 20:], dgs), axis=1)
  else:
    raise

  print('rewards on original dataset:', d.sum(), '/', len(d))

  with open(os.path.join(OUTPUT_FOLDER,'tr.pickle'), 'wb') as f:
    pickle.dump((s, a, r[:,None], ns, d[:,None]), f)



  # Make the augmented dataset for downstream task
  sas = np.concatenate([flat_sas_data]*2) # 20% real data

  FWD_MODEL = fwd_ma

  coda_parents = []
  for i in tqdm.tqdm(range(400)):
    coda_parents.append(FWD_MODEL(gmm_parents_model.generate_parents(10000)))
  coda_parents = np.concatenate(coda_parents)

  sas = np.concatenate((sas, coda_parents))
  dgs, r, d = reward_and_done_for_fetch(sas, use_target_goals=True, env=env, smaller_state=SMALL_STATE)
  if SMALL_STATE:
    s = np.concatenate((sas[:, :16], dgs), axis=1)
    a = sas[:, 16:20]
    ns = np.concatenate((sas[:, 20:], dgs), axis=1)

  print('rewards on augmented dataset:', d.sum(), '/', len(d))

  with open(os.path.join(OUTPUT_FOLDER,'coda.pickle'), 'wb') as f:
    pickle.dump((s, a, r[:,None], ns, d[:,None]), f)



  # Make the uniform augmented dataset for downstream task
  sas = np.concatenate([flat_sas_data]*2) # 20% real data

  OUTPUT_DATA_SIZE=5000000
  coda_parents = [flat_sas_data,flat_sas_data]

  while sum(map(len, coda_parents)) < OUTPUT_DATA_SIZE:
    print('.', end='')
    coda_parents.append(FWD_MODEL(prune_to_uniform(gmm_parents_model.generate_parents(30000), 10000.)))

  sas = np.concatenate(coda_parents)[:OUTPUT_DATA_SIZE]
  dgs, r, d = reward_and_done_for_fetch(sas, use_target_goals=True, env=env, smaller_state=SMALL_STATE)
  if SMALL_STATE:
    s = np.concatenate((sas[:, :16], dgs), axis=1)
    a = sas[:, 16:20]
    ns = np.concatenate((sas[:, 20:], dgs), axis=1)

  print('rewards on uniform augmented dataset:', d.sum(), '/', len(d))

  with open(os.path.join(OUTPUT_FOLDER,'coda_unif.pickle'), 'wb') as f:
    pickle.dump((s, a, r[:,None], ns, d[:,None]), f)
  
  plot_sample2(sas[-5000:], num_p=5000, filename=os.path.join(OUTPUT_FOLDER, 'coda_unif_dist.png'))

  print('All done!')
