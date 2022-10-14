"""
Trains batch RL agent on the fetch data

Running this file takes ????? minutes on a 1080 ti.

run with:

  batchrl_fetch.py --seed 0 --algo sac --dataset tr
  batchrl_fetch.py --seed 0 --algo bcddpg --dataset tr
  batchrl_fetch.py --seed 0 --algo sac --dataset coda
  batchrl_fetch.py --seed 0 --algo bcddpg --dataset coda
  batchrl_fetch.py --seed 0 --algo sac --dataset coda_unif
  batchrl_fetch.py --seed 0 --algo bcddpg --dataset coda_unif
  python batchrl_fetch.py --folder ./fetch_output_fc --seed 0 --dataset coda_unif --algo td3

outputs several files in 'fetch_output' subdirectory.
"""


import os, sys

from envs.customfetch.custom_fetch import FetchHookSweepAllEnv
from gym import Wrapper
from gym import spaces

from augment_offline_fetch import *
from mrl.import_all import *
from mrl.configs.make_continuous_agents import *

import torch
import numpy as np
import math
import pickle

import numpy as np
import torch
import torch.nn.functional as F

from mdn import MixtureDensityNetwork

class EnsembleMDN(mrl.Module):
  """
  An ensemble GMM model for modeling the empirical action distribution
  """
  def __init__(self, n_components = 5, optimize_every=5, batch_size=2000, hidden_dim=128, n_layers=2, ensemble_size=7, num_action_samples=20):
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
    self.num_action_samples = num_action_samples

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

  def forward(self, states):
    num_samples = self.num_action_samples
    div = num_samples // len(self.models) + 1
    inputs = torch.repeat_interleave(states, div, 0)
    outputs = torch.stack([model.sample(inputs) for model in self.models], 1) # batch * div, num_Models, out
    
    return outputs.reshape(states.shape[0], -1, self.env.action_dim)[:, :num_samples, :]
  
  def __call__(self, states):
    return self.forward(states)

  def save(self, save_folder: str):
    pass

  def load(self, save_folder: str):
    pass

class FlattenFetchHook(Wrapper):
  def __init__(self, env):
    super().__init__(env)
    assert isinstance(env, FetchHookSweepAllEnv)
    self.env = env
    assert self.env.smaller_state
    self.observation_space = spaces.Box(-np.inf, np.inf, shape=(20,), dtype='float32')
  
  def reset(self):
    state = self.env.reset()
    return np.concatenate((state['observation'], state['desired_goal'][[0,1,3,4]]))
        
  def step(self, action):
    next_state, reward, done, info = self.env.step(action)
    next_state = np.concatenate((next_state['observation'], next_state['desired_goal'][[0,1,3,4]]))
    return next_state, reward, done, info

if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser(description='Train Batch RL on Fetch Example')
  parser.add_argument('--dataset', type=str, default='tr', help='tr, coda, or coda_unif')
  parser.add_argument('--algo', type=str, default='sac', help='sac, bcddpg, or cql')
  parser.add_argument('--folder', type=str, default='./fetch_output', help='save folder for agent + results')
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--num_epochs', type=int, default=4010, help="50K steps / epoch @ 500 batch size")
  parser.add_argument('--num_action_samples', type=int, default=20, help="action_samples for bcddpg")
  parser.add_argument('--visualize_trained_agent', action='store_true', help="visualize the trained agent")
  args = parser.parse_args()


  import random
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)



  makeenv = lambda: FlattenFetchHook(FetchHookSweepAllEnv(place_two=True, smaller_state=True))
  e = makeenv()

  torch.set_num_threads(2)
  torch.set_num_interop_threads(2)

  if args.algo == 'sac':
    config = make_sac_agent(spinning_up_sac_config(), args=AttrDict(
      parent_folder=args.folder,
      env=makeenv,
      max_steps=75,
      replay_size=int(5e6)+1,
      alg='ddpg',
      layers=(512, 512, 512),
      clip_target_range=(-50, 0),
      gamma=0.98,
      tb=f'sac_{args.dataset}',
      epoch_len=1000,
      batch_size=500,
      log_every=100,
      num_envs=1,
      num_eval_envs=10,
      seed=args.seed
    ), agent_name_attrs=['alg', 'seed', 'tb'])
  
  elif args.algo == 'bcddpg':
    config = make_ddpg_agent(spinning_up_ddpg_config(), args=AttrDict(
      parent_folder=args.folder,
      env=makeenv,
      max_steps=75,
      replay_size=int(5e6)+1,
      alg='ddpg',
      layers=(512, 512, 512),
      clip_target_range=(-50, 0),
      gamma=0.98,
      tb=f'bcddpg_{args.dataset}',
      epoch_len=1000,
      batch_size=500,
      log_every=100,
      num_envs=1,
      num_eval_envs=10,
      seed=args.seed
    ), agent_name_attrs=['alg', 'seed', 'tb'])

    del config.module_algorithm
    config.module_algorithm = BCDDPG()
    del config.module_actor
    config.module_actor = EnsembleMDN(num_action_samples=args.num_action_samples)
    del config.module_policy
    config.module_policy = BatchConstrainedPolicy()

  elif args.algo == 'cql':
    config = make_sac_agent(spinning_up_sac_config(), args=AttrDict(
      parent_folder=args.folder,
      env=makeenv,
      max_steps=75,
      replay_size=int(5e6)+1,
      alg='cql',
      layers=(512, 512, 512),
      clip_target_range=(-50, 0),
      gamma=0.98,
      tb=f'cql_{args.dataset}',
      epoch_len=1000,
      batch_size=500,
      log_every=100,
      num_envs=1,
      num_eval_envs=10,
      seed=args.seed
    ), agent_name_attrs=['alg', 'seed', 'tb'])

    config.min_q_weight = 0.1
    del config.module_algorithm
    config.module_algorithm = SimpleConservativeSAC()

  elif args.algo == 'td3':
    config = make_td3_agent(spinning_up_td3_config(), args=AttrDict(
      parent_folder=args.folder,
      env=makeenv,
      max_steps=75,
      replay_size=int(5e6)+1,
      alg='td3',
      layers=(512, 512, 512),
      clip_target_range=(-50, 0),
      gamma=0.98,
      tb=f'td3_{args.dataset}',
      epoch_len=1000,
      batch_size=500,
      log_every=100,
      num_envs=1,
      num_eval_envs=10,
      seed=args.seed
    ), agent_name_attrs=['alg', 'seed', 'tb'])

    config.bc_loss = 1.0


  del config.module_replay
  config.module_replay = OldReplayBuffer()
  del config.module_state_normalizer

  agent  = mrl.config_to_agent(config)

  if args.visualize_trained_agent:
    print("Loading agent")
    agent.load('checkpoint_3900')
    agent.eval_mode()
    env = agent.env

    for _ in range(10000):
      state = env.reset()
      env.render()
      done = False
      while not done:
        time.sleep(0.02)
        action = agent.policy(state)
        state, reward, done, info = env.step(action)
        env.render()
      print(reward[0] > -0.3)
    quit()



  with open(os.path.join(args.folder, f'{args.dataset}.pickle'), 'rb') as f:
    dataset = pickle.load(f)

  agent.replay_buffer.buffer.add_batch(*dataset)

  if args.algo == 'bcddpg':
    # Train the S -> A density model for batch constrained policy
    agent.actor.optimize_every = 10000000000
    res = []
    for i in tqdm.tqdm(range(10000)):
      res.append(agent.actor._optimize(force=True))
    
    plt.figure()
    plt.plot(res)
    plt.savefig(os.path.join(agent.agent_folder, 'mdn_learning_curve.png'))

    # Cannot visualize the learned density model

  # Train agent in batch mode
  res = [np.mean(agent.eval(10).rewards)]
  for epoch in range(args.num_epochs):
    print(res[-1], end=' ')
    agent.train_mode()
    for _ in range(250):
      agent.config.env_steps +=1 # logger uses env_steps to decide when to write scalars.
      agent.config.opt_steps +=1
      agent.algorithm._optimize()
    agent.eval_mode()
    res += [np.mean(agent.eval(10).rewards)]
    if epoch % 50 == 0:
      agent.eval_mode()
      env = agent.eval_env

      collected_exps = [] # will be tuples of (s, a, s', r, d)

      while (len(collected_exps) < 200):
        state = env.reset()
        collected_exps.append(state)
        done = False
        n_steps = 0
        while n_steps < 75:
          n_steps += 1
          action = agent.policy(state)
          next_state, reward, done, info = env.step(action)
          state = next_state
          collected_exps.append(state)
      
      collected_exps = np.concatenate(collected_exps)
      plot_sample2(collected_exps, num_p=5000, filename=os.path.join(agent.agent_folder, f'trained_policy_viz_{epoch}.png'))
      print("Saving agent at epoch {}".format(epoch))
      agent.save(f'checkpoint_{epoch}')
