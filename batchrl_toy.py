"""
Trains batch RL agent on the toy data

Running this file takes <5 minutes on a 1080 ti.

run with:

  batchrl_toy.py --seed 0 --algo sac --dataset tr
  batchrl_toy.py --seed 0 --algo bcddpg --dataset tr
  batchrl_toy.py --seed 0 --algo sac --dataset coda
  batchrl_toy.py --seed 0 --algo bcddpg --dataset coda

outputs several files in 'toy_output' subdirectory.
"""


import os, sys

from augment_offline_toy import *
from mrl.import_all import *
from mrl.configs.make_continuous_agents import *

import torch
import numpy as np
import math
import pickle
import tqdm

import numpy as np
import torch
import torch.nn.functional as F

from mdn import MixtureDensityNetwork

class EnsembleMDN(mrl.Module):
  """
  An ensemble GMM model for modeling the empirical action distribution
  """
  def __init__(self, n_components = 5, optimize_every=5, batch_size=2000, hidden_dim=128, n_layers=2, ensemble_size=7):
    """
    Args:
      model_fn: Should be a constructor for an MLP that outputs N x [mean, std], so N x (2 x output_dims).
    """
    super().__init__('actor', required_agent_modules=['replay_buffer', 'env'], locals=locals())

    self.step = 0
    self.optimize_every = optimize_every
    self.batch_size = batch_size
    self.optimizer = None
    self.n_components = n_components
    self.hidden_dim = hidden_dim
    self.n_layers = n_layers
    self.ensemble_size = ensemble_size

  def _setup(self):
    self.models = [MixtureDensityNetwork(self.env.state_dim, self.env.action_dim, self.n_components, self.hidden_dim, self.n_layers)\
                   .to(self.config.device) for _ in range(self.ensemble_size)]
    params = sum([list(model.parameters()) for model in self.models], [])
    self.optimizer = torch.optim.Adam(params, lr=1e-3, weight_decay=5e-5)

  def _optimize(self, force=False):
    config = self.config
    self.step += 1

    if force or (self.step % self.optimize_every == 0 and len(self.replay_buffer) > 5000):
      sample = self.replay_buffer.buffer.sample(self.batch_size)
      states, actions = sample[0], sample[1]

      states = self.torch(states)
      actions = self.torch(actions)

      loss = sum([model.loss(states, actions) for model in self.models]) / len(self.models)
      loss = loss.mean()
      
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()

      return float(self.numpy(loss))

  def forward(self, states, num_samples=20):
    div = num_samples // len(self.models) + 1
    inputs = torch.repeat_interleave(states, div, 0)
    outputs = torch.stack([model.sample(inputs) for model in self.models], 1) # batch * div, num_Models, out
    
    return outputs.reshape(states.shape[0], -1, self.env.action_dim)[:, :num_samples, :]
  
  def __call__(self, states, num_samples=20):
    return self.forward(states, num_samples)

  def save(self, save_folder: str):
    pass

  def load(self, save_folder: str):
    pass


if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser(description='Train Batch RL on Toy Example')
  parser.add_argument('--dataset', type=str, default='tr', help='tr, coda, or coda_unif')
  parser.add_argument('--algo', type=str, default='sac', help='sac, bcddpg, or cql')
  parser.add_argument('--folder', type=str, default='./toy_output', help='save folder for agent + results')
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--num_epochs', type=int, default=350, help="50K steps / epoch @ 500 batch size")
  args = parser.parse_args()


  import random
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)


  makeenv = lambda: toyenv(mode=toymode.DIAG)
  e = makeenv()
  g = GroundTruthDynamics(e)
  fwd = lambda X: np.concatenate((X, g.forward(X[:,:2],X[:,2:])), 1)

  torch.set_num_threads(2)
  torch.set_num_interop_threads(2)

  if args.algo == 'sac':
    config = make_sac_agent(spinning_up_sac_config(), args=AttrDict(
      parent_folder=args.folder,
      env=makeenv,
      max_steps=70,
      replay_size=int(1e6)+1,
      alg='ddpg',
      layers=(512, 512),
      clip_target_range=(-50, 0),
      gamma=0.98,
      tb=f'sac_{args.dataset}',
      epoch_len=1000,
      batch_size=500,
      log_every=100,
      num_envs=1,
      seed=args.seed
    ), agent_name_attrs=['alg', 'seed', 'tb'])
  
  elif args.algo == 'bcddpg':
    config = make_ddpg_agent(spinning_up_ddpg_config(), args=AttrDict(
      parent_folder=args.folder,
      env=makeenv,
      max_steps=70,
      replay_size=int(1e6)+1,
      alg='ddpg',
      layers=(512, 512),
      clip_target_range=(-50, 0),
      gamma=0.98,
      tb=f'bcddpg_{args.dataset}',
      epoch_len=1000,
      batch_size=500,
      log_every=100,
      num_envs=1,
      seed=args.seed
    ), agent_name_attrs=['alg', 'seed', 'tb'])

    del config.module_algorithm
    config.module_algorithm = BCDDPG()
    del config.module_actor
    config.module_actor = EnsembleMDN()
    del config.module_policy
    config.module_policy = BatchConstrainedPolicy()

  elif args.algo == 'cql':
    config = make_sac_agent(spinning_up_sac_config(), args=AttrDict(
      parent_folder=args.folder,
      env=makeenv,
      max_steps=70,
      replay_size=int(1e6)+1,
      alg='cql',
      layers=(512, 512),
      clip_target_range=(-40, 0),
      gamma=0.98,
      tb=f'cql_{args.dataset}',
      epoch_len=1000,
      batch_size=500,
      log_every=100,
      num_envs=1,
      seed=args.seed
    ), agent_name_attrs=['alg', 'seed', 'tb'])

    config.min_q_weight = 0.1
    del config.module_algorithm
    config.module_algorithm = SimpleConservativeSAC()

  elif args.algo == 'td3':
    config = make_td3_agent(spinning_up_td3_config(), args=AttrDict(
      parent_folder=args.folder,
      env=makeenv,
      max_steps=70,
      replay_size=int(1e6)+1,
      alg='td3',
      layers=(512, 512),
      clip_target_range=(-50, 0),
      gamma=0.98,
      tb=f'td3_{args.dataset}',
      epoch_len=1000,
      batch_size=500,
      log_every=100,
      num_envs=1,
      seed=args.seed
    ), agent_name_attrs=['alg', 'seed', 'tb'])

    config.bc_loss = 1.0

  del config.module_replay
  config.module_replay = OldReplayBuffer()
  del config.module_state_normalizer

  agent  = mrl.config_to_agent(config)

  with open(os.path.join(args.folder, f'{args.dataset}{args.seed}.pickle'), 'rb') as f:
    dataset = pickle.load(f)

  agent.replay_buffer.buffer.add_batch(*dataset)

  if args.algo == 'bcddpg':
    # Train the S -> A density model for batch constrained policy
    agent.actor.optimize_every = 1000000000
    res = []
    for i in tqdm.tqdm_notebook(range(1000)):
      res.append(agent.actor._optimize(force=True))
    
    plt.figure()
    plt.plot(res)
    plt.savefig(os.path.join(agent.agent_folder, 'mdn_learning_curve.png'))

    # Visualize the learned density model

    random_states = np.random.uniform(size=(40000, 2))

    # visualizes the nth proposal only.
    n = 0
    D = np.concatenate((random_states, 
                        agent.numpy(agent.actor(agent.torch(random_states))[:,n,:])), 1)
    plot_sample_toy(fwd(D), 1000, filename=os.path.join(agent.agent_folder, 'trained_mdn_viz.png'))


  # Train agent in batch mode
  res = [np.mean(agent.eval(10).rewards)]
  for epoch in range(args.num_epochs):
    print(res[-1], end=' ')
    agent.train_mode()
    for _ in range(100):
      agent.config.env_steps +=1 # logger uses env_steps to decide when to write scalars.
      agent.config.opt_steps +=1
      agent.algorithm._optimize()
    agent.eval_mode()
    res += [np.mean(agent.eval(10).rewards)]
    if epoch % 50 == 0:
      random_states = np.random.uniform(size=(4000, 2))
      D = np.concatenate((random_states, agent.policy(random_states)), 1)
      plot_sample_toy(fwd(D), 1000, filename=os.path.join(agent.agent_folder, f'trained_policy_viz_{epoch}.png'))



  # Visualize the agent's policy
  random_states = np.random.uniform(size=(40000, 2))
  
  n = 0
  D = np.concatenate((random_states, agent.policy(random_states)), 1)
  plot_sample_toy(fwd(D), 1000, filename=os.path.join(agent.agent_folder, 'trained_policy_viz_final.png'))