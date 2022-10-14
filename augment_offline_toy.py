"""
Utility functions as well as full training code for the distributions/models on the TOY environment. 

Running this file takes <15 minutes on a 1080 ti.

run with:

  python augment_offline_toy.py --seed XXX

outputs several files in 'toy_output' subdirectory.
"""

import numpy as np
import os
from copy import deepcopy
import multiprocessing as mp
import matplotlib.pyplot as plt
import torch

from augment_offline import ConditionalGMMParentsModel, MLPDynamicsModel, MaskedMLPDynamicsModel
from augment_offline import make_graph, draw_graph, split_nodes

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

from gym import Env, spaces
from enum import Enum
from itertools import chain
import pickle
from IPython.display import clear_output


# TOY ENVIRONMENT

class toymode(Enum):
  UP = 0
  RIGHT = 1
  DIAG = 2
  UP_L = 3
  RIGHT_L = 4

class datamode(Enum):
  O = 0
  L = 1
  
class toyenv(Env):
  def __init__(self, mode=toymode.DIAG):
    super().__init__()
    
    self.observation_space = spaces.Box(np.zeros((2,)), np.ones((2,)), dtype=np.float32)
    self.action_space = spaces.Box(-np.ones((2,)), np.ones((2,)), dtype=np.float32)
    
    self.state = np.zeros((2,))
    
    self.mode = mode
    self.num_steps = 0
    
  def reset(self):
    self.num_steps = 0
    
    # by default starts the agent in bottom left
    self.state = np.random.random((2,)) * 0.15
    
    # if UP, starts the agent on the bottom, but not center
    if self.mode in [toymode.UP, toymode.UP_L]:
      if self.mode == toymode.UP_L or np.random.random() > 0.5:
        self.state[0] = np.random.random() * 0.4 + 0.6
      else:
        self.state[0] = np.random.random() * 0.4
    
    # if RIGHT, starts the agent on the left, but not center
    if self.mode in [toymode.RIGHT, toymode.RIGHT_L]:
      if self.mode == toymode.RIGHT_L or np.random.random() > 0.5:
        self.state[1] = np.random.random() * 0.4
      else:
        self.state[1] = np.random.random() * 0.4 + 0.6
      
    return self.state
  
  def render(self):
    
    plt.rcParams["figure.figsize"] = (3,3)

    fig1, ax1 = plt.subplots()

    ax1.set_ylim([0, 1])
    ax1.set_xlim([0, 1])
    ax1.add_patch(plt.Circle(self.state, 0.04, color='b'))
    ax1.add_patch(plt.Rectangle((0., 0.4), 1, 0.2, facecolor='r', alpha=0.1))
    ax1.add_patch(plt.Rectangle((0.4, 0.), 0.2, 1., facecolor='r', alpha=0.1))

    plt.show()
    clear_output(wait=True)
  
  def step(self, action):
    
    self.num_steps += 1
    
    if np.all(self.state > 0.5):
      self.state -= action[::-1] * 0.02
      
    self.state += action * 0.05
    
    reward = -1.
    done = self.num_steps > 70
    
    if self.mode in [toymode.UP, toymode.UP_L]:
      if self.state[1] > 0.9:
        reward = 0.
        done = True
    elif self.mode in [toymode.RIGHT, toymode.RIGHT_L]:
      if self.state[0] > 0.9:
        reward = 0.
        done = True
    elif self.mode == toymode.DIAG:
      if np.all(self.state > 0.8) and np.all(self.state < 0.9):
        reward = 0.
        done = True
    
    return self.state.copy(), reward, done, {}

def plot_sample_toy(sample, num_p=400, filename=None, errs=None, ax1=None):
 
  sample = sample[:num_p]

  if ax1 is None:
    plt.rcParams["figure.figsize"] = (8,8)
    fig1, ax1 = plt.subplots()
  ax1.set_ylim([0, 1])
  ax1.set_xlim([0, 1])
  ax1.add_patch(plt.Rectangle((0., 0.4), 1, 0.2, facecolor='r', alpha=0.05))
  ax1.add_patch(plt.Rectangle((0.4, 0.), 0.2, 1., facecolor='r', alpha=0.05))

  if errs is None:
    for s in sample:
      ax1.arrow(s[0],s[1],s[4]-s[0], s[5]-s[1], head_width=0.012, fc='b', ec='b', alpha=0.3)
  else:
    max_err = 0.1
    for s,e  in zip(sample,errs):
      ax1.arrow(s[0],s[1],s[4]-s[0], s[5]-s[1], head_width=0.012, fc='b', ec='b', alpha=min(e/max_err, 1.))


  if filename is not None:
    plt.savefig(filename)

TOY_NODES = [
   ('o1', {'idxs': [0], 'parents': []}),
   ('o2', {'idxs': [1], 'parents': []}),
   ('a1', {'idxs': [2], 'parents': []}),
   ('a2', {'idxs': [3], 'parents': []}),
   ('o1_2', {'idxs': [4], 'parents': ['o1','a1']}),
   ('o2_2', {'idxs': [5], 'parents': ['o2','a2']})
]

def collect_dataset_from_toy_env(size=40000, datamode=datamode.O):
  sample_dataset = []
  env = toyenv(mode=toymode.UP if datamode == datamode.O else toymode.UP_L)
  while len(sample_dataset) < size // 2:
    state = env.reset()
    done = False
    while not done:
      action = (np.random.random((2,)) * 0.1) - 0.05
      action[1] += np.random.random() * 0.5 + 0.4
      next_state, _, done, _ = env.step(action)
      #env.render()
      sample_dataset.append((state.copy(), action, next_state.copy()))
      state = next_state
    
  env = toyenv(mode=toymode.RIGHT if datamode == datamode.O else toymode.RIGHT_L)
  while len(sample_dataset) < size:
    state = env.reset()
    done = False
    while not done:
      action = (np.random.random((2,)) * 0.1) - 0.05
      action[0] += np.random.random() * 0.5 + 0.4
      next_state, _, done, _ = env.step(action)
      #env.render()
      sample_dataset.append((state.copy(), action, next_state.copy()))
      state = next_state

  return sample_dataset[:size]

class GroundTruthDynamics():
  
  def __init__(self, env):
    self.env = env
    
  def forward(self, s, a):
    """s here is the observation only, and a is the action. These can be batches"""
    if len(s.shape) == 1:
      s = s[None]
      a = a[None]
    
    res = []
    for state, action in zip(s, a):
      self.env.state = state.copy()
      next_state, _, _, _ = self.env.step(action)
      res.append(next_state.copy())
    
    return np.array(res)

# def generate_dummy_rc(roots,children):
#   """
#   Utility method that makes dummy roots and children for fully-connected graph
#   """
#   r, c = set(), set()
#   for child in children.values():
#     for p in child['parents']:
#       r.update(roots[p]['idxs'])
#     c.update(child['idxs'])
#   return {'r': list(r), 'c': list(c)}

def prune_to_uniform(proposals, target_size=12000.):
  from sklearn.neighbors import KernelDensity
  sample = proposals[-10000:]

  fmap = lambda s: s[:, :2]
  K = KernelDensity(bandwidth=0.05)
  K.fit(fmap(sample))
  scores = K.score_samples(fmap(proposals))
  scores = np.maximum(scores, np.log(0.01))
  scores = (1. / np.exp(scores))
  scores = scores / scores.mean()  * (target_size / len(proposals))
  
  return proposals[np.random.uniform(size=scores.shape) < scores]   

def generate_distributions(dataset, parent_set_idxs):
  full_data = dataset[np.random.permutation(range(len(dataset)))]
  tr = full_data[:-5000,:4]
  te = full_data[-5000:,:4]
  len(tr), len(te)

  M = ConditionalGMMParentsModel(n_components=32)
  with torch.no_grad():
    M.setup_parent_models(tr, parent_set_idxs)
    M.fit_parent_data(tr)

  mocoda_global = M.generate_parents(10000)
  #mocoda_disentangled = mocoda_global[np.logical_or(mocoda_global[:,0]<0.5,mocoda_global[:,1] < 0.5)]
  #mocoda_ent = mocoda_global[np.logical_not(np.logical_or(mocoda_global[:,0]<0.5,mocoda_global[:,1] < 0.5))]
  #_mocoda_ood = mocoda_global[np.logical_and(mocoda_global[:,0]<0.6,mocoda_global[:,1] < 0.6)]
  #mocoda_ood = _mocoda_ood[np.logical_and(_mocoda_ood[:,0]>0.4, _mocoda_ood[:,1]>0.4)]

  mocoda_global_uniform_pruned = prune_to_uniform(M.generate_parents(100000), 12000.)

  p_all = np.random.random((5000, 4))
  p_all[:,-2:] *= 2
  p_all[:,-2:] -= 1.

  return M, [
    tr,
    te,
    mocoda_global,
    #mocoda_disentangled,
    #mocoda_ent,
    #mocoda_ood,
    mocoda_global_uniform_pruned,
    p_all
  ]

def generate_dummy_rc(roots,children):
  r, c = set(), set()
  for child in children.values():
    for p in child['parents']:
      r.update(roots[p]['idxs'])
    c.update(child['idxs'])
  return {'r': {'idxs': list(r), 'parents': []}},\
        {'c': {'idxs': list(c), 'parents': ['r']}}



def train_fully_connected(base_model, roots, children, training_data, num_epochs=400, folder=None):

  model = MLPDynamicsModel(deepcopy(base_model), hidden_layers=[256,256])
  r_all, c_all = generate_dummy_rc(roots, children)
  model.setup_dynamics_models(training_data, r_all, c_all)

  tr_FC, val_FC = model.fit_dynamics(training_data, num_epochs)
  plot_training_curves(tr_FC, val_FC, 
    os.path.join(folder, 'learning_curve_fc.png') if folder else None)

  fwd_fc = lambda X: np.concatenate((X, model.generate_children(X)), 1)

  return model, fwd_fc

def train_disentangled(base_model, roots, children, training_data, num_epochs=400, folder=None):
  model = MLPDynamicsModel(deepcopy(base_model), hidden_layers=[256,256])
  model.setup_dynamics_models(training_data, roots, children)
  tr_DS, val_DS = model.fit_dynamics(training_data, num_epochs)
  plot_training_curves(tr_DS, val_DS, 
    os.path.join(folder, 'learning_curve_ds.png') if folder else None)
  fwd_ds = lambda X: np.concatenate((X, model.generate_children(X)), 1)

  return model, fwd_ds

def train_fc_masked(base_model, roots, children, training_data, num_epochs=400, folder=None):
  # emulate fully connected w/ Masked Network
  r_all, c_all = generate_dummy_rc(roots, children)
  model = MaskedMLPDynamicsModel(deepcopy(base_model), hidden_layers=[256,256])
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

def train_ds_masked(base_model, roots, children, training_data, num_epochs=400, folder=None):
  # emulate fully connected w/ Masked Network
  model = MaskedMLPDynamicsModel(deepcopy(base_model), hidden_layers=[256,256])
  def fc_mask_fn(input_tensor):
    """
    accepts B x total_in_dim and produces B x num_roots x num_children
    returns all 1
    """
    res = torch.tensor([[1, 0],[0, 1],[1, 0],[0, 1]]).to(input_tensor.device)
    
    return res[None].repeat((input_tensor.shape[0], 1, 1))

  model.setup_dynamics_models(training_data, roots, children, fc_mask_fn)

  tr_MA, val_MA = model.fit_dynamics(training_data, num_epochs)
  plot_training_curves(tr_MA, val_MA, 
    os.path.join(folder, 'learning_curve_mads.png') if folder else None)

  fwd_ma = lambda X: np.concatenate((X, model.generate_children(X)), 1)

  return model, fwd_ma

def train_toy_masked(base_model, roots, children, training_data, num_epochs=400, folder=None):
  # emulate fully connected w/ Masked Network
  model = MaskedMLPDynamicsModel(deepcopy(base_model), hidden_layers=[256,256])
  def fc_mask_fn(input_tensor):
    """
    accepts B x total_in_dim and produces B x num_roots x num_children
    returns all 1
    """
    res = torch.tensor([[1, 0],[0, 1],[1, 0],[0, 1]]).to(input_tensor.device)
    res = res[None].repeat((input_tensor.shape[0], 1, 1))
    res[torch.logical_and(input_tensor[:,0] > 0.5, input_tensor[:, 1] > 0.5)] = 1
    
    return res

  model.setup_dynamics_models(training_data, roots, children, fc_mask_fn)

  tr_MA, val_MA = model.fit_dynamics(training_data, num_epochs)
  plot_training_curves(tr_MA, val_MA, 
    os.path.join(folder, 'learning_curve_ma.png') if folder else None)

  fwd_ma = lambda X: np.concatenate((X, model.generate_children(X)), 1)

  return model, fwd_ma


def plot_dists(DISTS, fwd, saveas=None):
  #[print(d.shape) for d in DISTS]
  plt.rcParams["figure.figsize"] = (20,8)
  fig, axes = plt.subplots(2, 5)

  for ax, d in zip(chain.from_iterable(axes), DISTS):
    plot_sample_toy(fwd(d), 1000, ax1=ax)
  
  if saveas:
    fig.savefig(saveas)

def plot_training_curves(tr, val, saveas=None):
  plt.rcParams["figure.figsize"] = (4, 3)
  plt.figure()
  plt.plot(tr)
  plt.plot(val)

  if saveas:
    plt.savefig(saveas)

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
    plot_sample_toy(model(dist), 1500, errs=errs, ax1=ax)

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

def reward_and_done_for_toy(sas, mode='top_right'):
  states = sas[:, -2:]
  rs = []
  if mode == 'top_right':
    for s in states:
      if np.all(s > 0.8) and np.all(s < 0.9):
        rs.append(0.)
      else:
        rs.append(-1.)
  elif mode == 'top_left':
    for s in states:
      if (s[0] > 0.1) and (s[1] > 0.8) and (s[0] < 0.2) and (s[1] < 0.9):
        rs.append(0.)
      else:
        rs.append(-1.)

  return np.array(rs)[:,None], (np.array(rs)==0.)[:,None]

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description='Mocoda on Toy Example')
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--output_folder', type=str, default='./toy_output')
  parser.add_argument('--datamode', type=str, default='L')
  args = parser.parse_args()
  
  import random
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)

  OUTPUT_FOLDER = args.output_folder
  BASE_DATA_SIZE = 40000
  OUTPUT_DATA_SIZE = 200000

  # Make output folder if it doesn't exist
  if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
  
  # Collect 40K samples from Environment
  DATAMODE = datamode.O if args.datamode == "O" else datamode.L
  flat_sas_data = np.stack(list(map(np.concatenate, collect_dataset_from_toy_env(BASE_DATA_SIZE, DATAMODE)))) # 40k x 6
  flat_sas_data = flat_sas_data[np.random.permutation(len(flat_sas_data))]

  
  # Get ground truth dynamics model
  g = GroundTruthDynamics(toyenv())
  fwd = lambda X: np.concatenate((X, g.forward(X[:,:2],X[:,2:])), 1)

  # Construct Causal Graph and save SCM diagram
  G = make_graph(TOY_NODES)
  draw_graph(G, os.path.join(OUTPUT_FOLDER, 'global_causal_graph.png'))

  roots, children = split_nodes(G)
  parent_sets = [child['parents'] for child in children.values()]
  parent_set_idxs = []
  for parent_set in parent_sets:
    parent_set_idxs.append(np.concatenate([roots[p]['idxs'] for p in parent_set]))



  # Generate and plot the distributions
  gmm_parents_model, DISTS = generate_distributions(flat_sas_data, parent_set_idxs)
  tr, te, mocoda_global, mocoda_uniform_pruned, p_all = DISTS

  plot_dists(DISTS, fwd, os.path.join(OUTPUT_FOLDER, 'distributions.png'))

  #  Train all of the models on the training distribution
  C_FC, fwd_fc = train_fully_connected(gmm_parents_model, roots, children, fwd(tr), num_epochs=600, folder=OUTPUT_FOLDER)
  C_DS, fwd_ds = train_disentangled(gmm_parents_model, roots, children, fwd(tr), num_epochs=600, folder=OUTPUT_FOLDER)
  #C_MAFC, fwd_mafc = train_fc_masked(gmm_parents_model, roots, children, fwd(tr), num_epochs=600, folder=OUTPUT_FOLDER)
  #C_MADS, fwd_mads = train_ds_masked(gmm_parents_model, roots, children, fwd(tr), num_epochs=600, folder=OUTPUT_FOLDER)
  C_MA, fwd_ma = train_toy_masked(gmm_parents_model, roots, children, fwd(tr), num_epochs=600, folder=OUTPUT_FOLDER)

  #named_models = [('fc', fwd_fc), ('ds', fwd_ds), ('mafc', fwd_mafc), ('mads', fwd_mads), ('ma', fwd_ma)]
  named_models = [('fc', fwd_fc), ('ds', fwd_ds), ('ma', fwd_ma)]


  def random_rollouts(base_data, steps=5, model=C_FC):
    new_data = []
    current_data = base_data
    for i in range(steps):
      s = model.generate_children(current_data)
      current_data = np.concatenate((s, np.random.random(s.shape) * 2 - 1), 1)
      current_data = np.concatenate((current_data, model.generate_children(current_data)), 1)
      new_data.append(current_data)
    return np.concatenate(new_data, 0)

  base_dist = flat_sas_data[:10000]
  new_dist = random_rollouts(base_dist)
  dyna_dist = np.concatenate((base_dist, new_dist))[:,:4]

  named_dists = list(zip([DISTS[1], dyna_dist, DISTS[2], DISTS[3], DISTS[4]], ["te", "dyna", "mocoda_global", "mocoda_uniform_pruned", "p_all"]))

  # Compare the models across all distributions
  compare_models_on_dists(named_models, named_dists, OUTPUT_FOLDER, args.seed)
  
  # Make the augmented dataset for downstream task

  # CODA
  coda_data = np.zeros((OUTPUT_DATA_SIZE - BASE_DATA_SIZE, 6))

  coda_data[:, [0, 2, 4]] = flat_sas_data[np.random.choice(len(flat_sas_data), OUTPUT_DATA_SIZE - BASE_DATA_SIZE)][:, [0, 2, 4]]
  coda_data[:, [1, 3, 5]] = flat_sas_data[np.random.choice(len(flat_sas_data), OUTPUT_DATA_SIZE - BASE_DATA_SIZE)][:, [1, 3, 5]]
  raw_data = np.concatenate((flat_sas_data[:BASE_DATA_SIZE], coda_data))
  s = raw_data[:, :2]
  a = raw_data[:, 2:4]
  ns = raw_data[:, 4:]
  r, d = reward_and_done_for_toy(raw_data)

  print('rewards on coda dataset:', d.sum(), '/', len(d))

  with open(os.path.join(OUTPUT_FOLDER, f'ogcoda_{args.seed}.pickle'), 'wb') as f:
    pickle.dump((s, a, r, ns, d), f)

  # MOCODA

  # Make the base dataset for downstream task
  raw_data = flat_sas_data[np.random.choice(len(flat_sas_data), OUTPUT_DATA_SIZE)]
  s = raw_data[:, :2]
  a = raw_data[:, 2:4]
  ns = raw_data[:, 4:]
  r, d = reward_and_done_for_toy(raw_data)

  print('rewards on base dataset:', d.sum(), '/', len(d))

  with open(os.path.join(OUTPUT_FOLDER, f'tr{args.seed}.pickle'), 'wb') as f:
    pickle.dump((s, a, r, ns, d), f)


  # Make the dyna dataset for downstream task
  mocoda_parents = [flat_sas_data[:40000]]

  while sum(map(len, mocoda_parents)) < OUTPUT_DATA_SIZE:
    mocoda_parents.append(random_rollouts(flat_sas_data[:40000]))

  raw_data = np.concatenate(mocoda_parents)[:OUTPUT_DATA_SIZE]
  s = raw_data[:, :2]
  a = raw_data[:, 2:4]
  ns = raw_data[:, 4:]
  r, d = reward_and_done_for_toy(raw_data)

  print('rewards on augmented dataset:', d.sum(), '/', len(d))

  with open(os.path.join(OUTPUT_FOLDER, f'dyna{args.seed}.pickle'), 'wb') as f:
    pickle.dump((s, a, r, ns, d), f)


  # Make the augmented dataset for downstream task
  mocoda_parents = gmm_parents_model.generate_parents(OUTPUT_DATA_SIZE - BASE_DATA_SIZE)
  raw_data = np.concatenate((flat_sas_data[:40000], fwd_ma(mocoda_parents)))
  s = raw_data[:, :2]
  a = raw_data[:, 2:4]
  ns = raw_data[:, 4:]
  r, d = reward_and_done_for_toy(raw_data)

  print('rewards on augmented dataset:', d.sum(), '/', len(d))

  with open(os.path.join(OUTPUT_FOLDER, f'mocoda{args.seed}.pickle'), 'wb') as f:
    pickle.dump((s, a, r, ns, d), f)


  # Make the uniform augmented dataset for downstream task
  mocoda_parents = [flat_sas_data[:40000]]

  while sum(map(len, mocoda_parents)) < OUTPUT_DATA_SIZE:
    mocoda_parents.append(fwd_ma(prune_to_uniform(gmm_parents_model.generate_parents(100000), 12000.)))

  raw_data = np.concatenate(mocoda_parents)[:OUTPUT_DATA_SIZE]
  s = raw_data[:, :2]
  a = raw_data[:, 2:4]
  ns = raw_data[:, 4:]
  r, d = reward_and_done_for_toy(raw_data)

  print('rewards on augmented dataset:', d.sum(), '/', len(d))

  with open(os.path.join(OUTPUT_FOLDER, f'mocoda_unif{args.seed}.pickle'), 'wb') as f:
    pickle.dump((s, a, r, ns, d), f)

  
  # Make the random augmented dataset for downstream task  
  mocoda_parents = np.random.random((OUTPUT_DATA_SIZE - BASE_DATA_SIZE, 4))
  mocoda_parents[:,-2:] *= 2
  mocoda_parents[:,-2:] -= 1.

  raw_data = np.concatenate((flat_sas_data[:40000], fwd_ma(mocoda_parents)))
  s = raw_data[:, :2]
  a = raw_data[:, 2:4]
  ns = raw_data[:, 4:]
  r, d = reward_and_done_for_toy(raw_data)

  print('rewards on augmented dataset:', d.sum(), '/', len(d))

  with open(os.path.join(OUTPUT_FOLDER, f'rand{args.seed}.pickle'), 'wb') as f:
    pickle.dump((s, a, r, ns, d), f)