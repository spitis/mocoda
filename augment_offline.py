from collections import defaultdict
import copy
from itertools import chain
from math import ceil
from pprint import pprint

import matplotlib.pyplot as plt
from munch import Munch
import networkx as nx
import numpy as np
from torch._C import device
from torch.distributions import pareto
import tqdm
import math
import gc

import torch
import torch.nn as nn
import torch.distributions as D
from torch.utils.data import TensorDataset, DataLoader

from gmm_torch import GaussianMixture as GMM, pairwise_chernoff_divergences, mixture_entropy_estimator_with_pairwise_distances, diag_gaussian_batch_entropies, conditional_entropy_gmm
from density_utils import FeaturewiseKDE, StaticGMM
from cond_gmm import condition_gmm

def make_graph(nodes):
  """
  Make a networkx graph representing the causal model.

  The input "nodes" should be of type:

    [('node_name' : {'idxs': list_of_int, 'parents': [list_of_node_names]})]

  the idxs index a flattened (s, a, s') tuple.

  Example input 'nodes':

  nodes = [('pos_1', {'idxs': [0, 1], 'parents': []}),
            ('ob1_1', {'idxs': [2], 'parents': []}),
            ('ob2_1', {'idxs': [3], 'parents': []}),
            ('act_pos', {'idxs': [4, 5], 'parents': []}),
            ('act_ob1', {'idxs': [6], 'parents': []}),
            ('act_ob2', {'idxs': [7], 'parents': []}),
            ('pos_2', {'idxs': [8, 9], 'parents': ['pos_1', 'act_pos']}),
            ('ob1_2', {'idxs': [10], 'parents': ['pos_1', 'ob1_1', 'act_ob1']}),
            ('ob2_2', {'idxs': [11], 'parents': ['pos_1', 'ob2_1', 'act_ob2']})]

  """
  G = nx.DiGraph()
  G.add_nodes_from(nodes)
  for node in G.nodes:
    for parent in G.nodes[node]['parents']:
      G.add_edge(parent, node)
  return G


def draw_graph(G, saveas=None):
  """
  Draws the bipartite causal graph G, where G is a networkx repr. created by "make_graph"
  """
  options = {
      "font_size": 10,
      "node_size": 2000,
      "node_color": "white",
      "edgecolors": "black",
      "linewidths": 3,
      "width": 3,
  }
  plt.rcParams["figure.figsize"] = (4, 6)

  parents = [n for n in G.nodes if len(G[n])]

  nx.draw_networkx(G, pos=nx.bipartite_layout(G, parents), **options)
  # nx.draw_networkx(G, pos=nx.bipartite_layout(G, [r'$p_1$', 'ob1_1', 'ob2_1', 'act_pos', 'act_ob1', 'act_ob2']), **options)

  # TODO(): fill in nodes with latex symbols
  #nx.draw_networkx(G, pos=nx.bipartite_layout(G, ['pos_1', 'ob1_1', 'ob2_1', 'ob3_1', 'act_pos', 'act_ob1', 'act_ob2', 'act_ob3']), **options)

  # Set margins for the axes so that nodes aren't clipped
  ax = plt.gca()
  ax.margins(x=0.20)
  plt.axis("off")
  plt.tight_layout()

  if saveas:
    plt.savefig(saveas)


def split_nodes(G):
  """
  Splits graph into root nodes (no parents) and child nodes.
  """
  root = {}
  children = {}
  for node in G.nodes:
    node_ = G.nodes[node]
    if len(node_['parents']):
      children[node] = node_
    else:
      root[node] = node_

  return root, children


def connected_components(G):
  """
  splits Graph into its connected components.
  """
  return list(nx.connected_components(G.to_undirected()))


def data_to_nodes(data, nodes):
  """
  Transforms a (s, a, s', r, d) data tuple into a node dictionary.
  """
  sas = np.concatenate(data[:3])
  return {k: sas[v['idxs']] for k, v in nodes.items()}


def flatten_dataset(dataset):
  """
  Transforms dataset of (s, a, s', r, d) tuples into concatentated (s, a, s') array.
  """
  rows = [np.concatenate(data[:3]) for data in dataset]
  return np.stack(rows)


def filter_by_constraint(flat_sas_data, idxs, value, epsilon=0.05):
  """
  Filters a flattened dataset according to a constraint on some of its indices.
  E.g., to find points where the agent is near a given position.
  """
  satisfies_constraint = np.linalg.norm(flat_sas_data[:, idxs] - np.array(value)[None], axis=1) < epsilon
  return flat_sas_data[satisfies_constraint]


def naive_root_sampler(flattened_dataset, roots, children, rng=np.random):
  """
  This is an empirical sampler of the root nodes. It fills the sampled_roots with empirical samples of one
  parent set a time, ensuring that any parents shared with prior parent_set samples are consistent.
  """
  sampled_roots = {}
  parent_sets = [child['parents'] for child in children.values()]
  for parent_set in rng.permutation(np.array(parent_sets, dtype='object')):
    dataset = flattened_dataset.copy()
    parents_to_add = []
    for parent in parent_set:
      if parent in sampled_roots:
        dataset = filter_by_constraint(dataset, roots[parent]['idxs'], sampled_roots[parent])
      else:
        parents_to_add.append(parent)
    sampled_point = dataset[rng.randint(len(dataset))]
    for parent in parents_to_add:
      sampled_roots[parent] = sampled_point[roots[parent]['idxs']]
  # NOTE: the returned rng can be used for sampling the children
  return sampled_roots, rng


def naive_child_sampler(flattened_dataset, root_sample, roots, children, rng=np.random):
  """
  This is an empirical sampler of the child nodes. It returns a complete state given some root sample,
  by sampling children from the entire dataset, such that their empirical parents are "close enough" to
  their parents in the root node.

  IMPORTANT: Returns NONE if there are no "close enough" parents in the dataset.
  """
  sample = copy.copy(root_sample)
  for childk, childv in children.items():
    dataset = flattened_dataset.copy()
    query = np.concatenate([sample[root] for root in childv['parents']])
    query_idxs = np.concatenate([roots[root]['idxs'] for root in childv['parents']])
    dataset = filter_by_constraint(dataset, query_idxs, query, epsilon=0.025)
    if not len(dataset):
      return None
    sampled_point = dataset[rng.randint(len(dataset))]
    sample[childk] = sampled_point[childv['idxs']]
  return sample


def naive_sampler(flattened_dataset, roots, children, rng=np.random):
  """
  This joins together the naive_root_sampler and naive_child_sampler to
  naively sample ONE (s, a, s') tuple (formatted as a node dict).
  Either component might be replaced later on.
  """
  sample, rng = naive_root_sampler(flattened_dataset, roots, children, rng=rng)
  return naive_child_sampler(flattened_dataset, sample, roots, children, rng=rng)


def flatten_sample(sas_size, sample, roots, children):
  """
  flattens a dict sample to a concatenated (sas) vector.
  """
  flat = np.zeros((sas_size, ), dtype=np.float32)
  for rootk, rootv in roots.items():
    flat[rootv['idxs']] = sample[rootk]
  for childk, childv in children.items():
    flat[childv['idxs']] = sample[childk]
  return flat


def generate_naive_coda_dataset(max_N, flattened_dataset, roots, children, sampler=naive_sampler, rng=np.random):
  """
  generates (N x (s, a, s')) array using empirical sampling
  """
  samples = []
  sample_len = flattened_dataset.shape[1]
  for i in tqdm.tqdm(range(max_N)):
    sample = sampler(flattened_dataset, roots, children, rng=rng)
    if sample is not None:
      samples.append(flatten_sample(sample_len, sample, roots, children))
  return np.stack(samples)


def generate_gmm_coda_dataset(args, flattened_dataset, roots, children, sampler=naive_sampler, rng=np.random):
  """
  generates (N x (s, a, s')) array by sampling from trained generative models

  This is only meant to be called once (i.e. not parallelized) since it trains the models on each call
  """
  # TODO(): add GPU/CUDA support

  ##############################################################################
  # flags
  cfg = Munch()
  cfg.num_components = 5  #@param {type:"number"}
  cfg.num_epochs = 800  #@param {type:"number"}
  cfg.batch_size = 5000  #@param {type:"number"}
  cfg.lr = 1e-5  #@param {type:"number"}
  cfg.weight_decay = 5e-5  #@param {type:"number"}
  cfg.std_activation = 'exp'
  cfg.min_std = 1e-4  #@param {type:"number"}
  cfg.hidden_layer_sizes = [256]
  cfg.fit_shared_root = True
  cfg.plot = True  # produce plots to qualitatively examine goodness-of-fit
  # TODO(): add these as command-line flags to augment_offline_impl.py?
  print('Generating GMM CoDA dataset with the following config:')
  pprint(cfg)

  ######################################################################ht -0########

  ##############################################################################
  # data
  def factored_roots(roots, children):
    """Determine which children share roots and use this info to factor the roots."""

    # NOTE: not sure this will gracefully handle > 1 shared root variable...

    def intersection(list_of_lists):
      """Util for determining shared parents from the parent_sets"""
      set_ = set(list_of_lists[0])  # start with first one
      for l in list_of_lists[1:]:
        set_ = set_.intersection(set(l))  # keep unioning
      return list(set_)

    parent_sets = [child['parents'] for child in children.values()]
    shared_roots = intersection(parent_sets)
    factored_parent_sets = defaultdict(list)
    # factored_parent_sets maps each conditioning root variable to a set of
    # other root varibles in a factor e.g. for the 2-D problem where we have root
    # variables that factor as
    #   P(pos, obj_1, obj_2, act_pos, act_obj_1, act_obj_2) =
    #   P(pos)P(act_pos|pos)P(obj_1, act_obj_1|pos)P(obj_1, act_obj_1|pos)
    # this data structure maps "pos" to a list of its dependent factors, i.e.
    #   [[act_pos], [obj_1, act_obj_1], [obj_2, act_obj_2]]
    for root in shared_roots:
      for parent_set in parent_sets:
        parent_set_besides_root = set(parent_set).difference([root])
        factored_parent_sets[root].append(list(parent_set_besides_root))
    # shared_parents_ = {k: v for k, v in roots.items() if k in shared_roots}
    # return shared_parents_
    return factored_parent_sets

  factored_roots_ = factored_roots(roots, children)  # should be pos_1
  shared_root = next(iter(factored_roots_.keys()))  # assumes only one shared root
  ##############################################################################

  ##############################################################################
  # fit generative models
  root_gmms = dict()
  child_gmms = dict()

  from gmm_utils import fit_cond_gmm_mlp
  # fit generative models for the root dist'n (leveraging how it factors)
  #   start by (optionally) fitting marginal over shared_root
  if cfg.fit_shared_root:
    from gmm_utils import fit_marg_gmm_mlp
    idx_inp = roots[shared_root]['idxs']
    print()
    print(f'Fitting P({shared_root})')
    print(f'  Target variable ({shared_root}) has {len(idx_inp)} dimensions')
    root_gmms[shared_root] = fit_marg_gmm_mlp(flattened_dataset, idx_inp, cfg)
  #   continue by fitting remaining root factors p(roots_i|shared roots) for all i
  for target_roots in factored_roots_[shared_root]:
    conditioning_root = shared_root
    key = ','.join(target_roots) + '|' + conditioning_root
    idx_inp = roots[shared_root]['idxs']
    idx_out = []
    for root in target_roots:
      idx_out.extend(roots[root]['idxs'])
    print()
    print(f'Fitting P({key})')
    print(f'  Target variable ({key}) has idx={idx_out} ({len(idx_out)} dimensions)')
    print(f'  Conditioning variable ({shared_root}) has idx={idx_inp} ({len(idx_inp)} dimensions)')
    root_gmms[key] = fit_cond_gmm_mlp(flattened_dataset, idx_inp, idx_out, cfg)
  # fit generative models for the children|root dist'ns - one per SCM
  for child, child_properties in children.items():
    parents = child_properties['parents']
    parents_str = ','.join(parents)
    idx_inp = [roots[parent]['idxs'] for parent in parents]
    idx_inp = sum(idx_inp, [])  # flatten list of lists
    idx_out = child_properties['idxs']
    key = f'{child}|{parents_str}'
    print()
    print(f'Fitting P({key})')
    print(f' Target variable ({child}) has idx={idx_out} ({len(idx_out)} dimensions)')
    print(f' Conditioning variable ({parents_str}) has idx={idx_inp} ({len(idx_inp)} dimensions)')
    child_gmms[key] = fit_cond_gmm_mlp(flattened_dataset, idx_inp, idx_out, cfg)

  if cfg.plot:
    import os
    DIRNAME = '.' if args is None else os.path.dirname(args.target_hdf5)
    DIRNAME = os.path.join(DIRNAME, 'figs')
    os.makedirs(DIRNAME, exist_ok=True)
    from gmm_utils import plot_learning_curve
    from gmm_utils import plot_gmm_marginals
    from gmm_utils import scatter_plot_gmm_samps
    from gmm_utils import ConditionalGMM
    for gmm_dict in (root_gmms, child_gmms):
      for name, gmm in gmm_dict.items():
        basename = name
        filename = f'{DIRNAME}/{basename}-learning.png'
        plot_learning_curve(gmm, filename)
        filename = f'{DIRNAME}/{basename}-marginals.png'
        plot_gmm_marginals(gmm, flattened_dataset, filename, args)
        # scatter plot samps if the model is fit to 2d data
        if isinstance(gmm, ConditionalGMM):
          num_data_dim = len(gmm.idx_out)
        else:  # Marginal GMM
          num_data_dim = len(gmm.idx_inp)
        if num_data_dim == 2:
          filename = f'{DIRNAME}/{basename}-scatter.png'
          scatter_plot_gmm_samps(gmm, flattened_dataset, filename, args)
  ##############################################################################

  ##############################################################################
  # Ancestral sampling

  # batch generator compatable with total_samples > len(flattened_dataset)
  num_complete_batches, leftover = divmod(args.total_samples, cfg.batch_size)
  num_batches = num_complete_batches + bool(leftover)

  def data_stream():
    while True:
      perm = rng.permutation(len(flattened_dataset))
      for i in range(num_batches):
        idx_sta = (i * cfg.batch_size) % len(flattened_dataset)  # start here
        idx_end = ((i + 1) * cfg.batch_size) % len(flattened_dataset)  # end here
        if idx_end < idx_sta:  # last batch - time to wrap around
          batch_idx = np.hstack((
              perm[idx_sta:],
              perm[:idx_end],
          ))
        else:
          batch_idx = perm[idx_sta:idx_end]
        yield flattened_dataset[batch_idx]

  batches = data_stream()

  coda_batches = []
  t = torch.tensor
  n = np.array
  # NOTE: when drawing conditional GMM samples we pass the entire batch to the GMM,
  #       and any required masking/conditioning happens within the GMM forward method
  for _ in range(num_batches):
    batch = next(batches)
    coda_batch = np.zeros_like(batch)
    kwargs = dict(sample=not args.test_time_means)
    # sample shared root
    if cfg.fit_shared_root:  # sample from the model for P(shared_soot)
      gmm = root_gmms[shared_root]
      samp = gmm(t(batch), **kwargs)
    else:  # sample shared_root empirically from batch data
      samp = batch[:, gmm.idx_inp]
    coda_batch[:, gmm.idx_inp] = n(samp)  # use idx_inp when marginal sampling
    # sample remaining roots conditioned on shared root
    for key, gmm in root_gmms.items():
      if key == shared_root:  # sampled this one already
        continue
      # we condition on the previous shared_root value to sample remaining roots
      samp = gmm(t(coda_batch), **kwargs)  # NOTE: only shared_root dimensions will be conditioned on
      coda_batch[:, gmm.idx_out] = n(samp)
    # sample next-state childen conditioned on parent-roots
    for key, gmm in child_gmms.items():
      samp = gmm(t(coda_batch), **kwargs)  # NOTE: only parent-root dimensions will be conditioned on
      coda_batch[:, gmm.idx_out] = n(samp)
    coda_batches.append(coda_batch)
  samples = np.vstack(coda_batches)
  samples = samples[:args.total_samples]  # return *exactly* the number of samples requested
  ##############################################################################

  if args.round:  # round dimensions of the GMM samples that make sense to round
    #
    # helpers to determine which input data are unique (we round these dims only)
    def _num_unique_values_per_dim(dim):
      """Count the number of unqiue values in dim of flattened_dataset."""
      subsample_data = flattened_dataset[:, dim]
      _, counts = np.unique(subsample_data, return_counts=True)
      return len(counts)

    #
    def _quantize_dim(dim):
      """Quantize dim of samples using unqiue values from flattened_dataset."""
      bins = np.unique(flattened_dataset[:, dim])
      nearest_bins = np.argmin(np.abs(samples[:, dim, None] - bins), axis=1)
      return bins[nearest_bins]

    #

    # for each dim, quantize to observed values if this dim had discrete values
    _, num_dims = flattened_dataset.shape
    for i in range(num_dims):
      # NOTE: arbitrarily assume discrete rand vars take on more than 5 values
      is_discrete = _num_unique_values_per_dim(i) < 5
      if is_discrete:
        samples[:, i] = _quantize_dim(i)
    #

  return samples


from abc import ABC, abstractmethod, abstractproperty


class ParentsModel(ABC):

  @abstractproperty
  def name(self):
    pass

  @abstractmethod
  def setup_parent_models(self, parent_data, parent_set_idxs):
    """
    Sets up the models. Called before fit methods.
    """
    pass

  @abstractmethod
  def fit_parent_data(self, parent_data):
    """
    Should fit the parent_data to match the marginals/support of each parent_set,
      while maximizing entropy/matching a target distribution.
    """
    pass

  @abstractmethod
  def generate_parents(self, n_samples):
    """
    Should generate a numpy array of shape (n_samples, parent_dims). 
    """
    pass

class DynamicsModel(ParentsModel, ABC):

  def __init__(self, parents_model):
    self.parents_model = parents_model

  @abstractmethod
  def name(self):
    pass

  def setup_parent_models(self, *args, **kwargs):
    return self.parents_model.setup_parent_models(*args, **kwargs)

  def fit_parent_data(self, *args, **kwargs):
    return self.parents_model.fit_parent_data(*args, **kwargs)

  def generate_parents(self,  *args, **kwargs):
    return self.parents_model.generate_parents(*args, **kwargs)

  @abstractmethod
  def setup_dynamics_models(self, flat_sas_data, roots, children):
    """
    Should set up the dynamics models.
    """
    pass

  @abstractmethod
  def fit_dynamics(self, flat_sas_data):
    """
    Should fit the dynamics of the full dataset using a disentangled dynamics model.
    """
    pass

  @abstractmethod
  def generate_children(self, parents):
    """
    Should generate a numpy array of shape (len(parents), child_dims). 
    """
    pass

  def generate_data(self, n_samples):
    assert(hasattr(self, 'generate_children'))
    parents = self.generate_parents(n_samples)
    children = self.generate_children(parents)
    return np.concatenate((parents, children), -1)

class GlobalGMMParentsModel(ParentsModel):
  """
  This model fits the parent distribution using a global GMM model that is optimized
    using a disentangled loss function.
  """
  def __init__(self, n_components=512, batch_size=512, n_workers=4, device='cuda'):
    self.n_components = n_components
    self.batch_size = batch_size
    self.n_workers = n_workers
    self.model = None
    self.opt = None
    self.device = device
    self.parent_sample = None
    self.parent_set_idxs = None

  def name(self):
    return 'GlobalGMM'

  def setup_parent_models(self, parent_data, parent_set_idxs, lr=2e-3):
    s1 = torch.Tensor(parent_data)[:100000].to(self.device)
    self.parent_sample = s1

    self.model = GMM(self.n_components, parent_data.shape[1], covariance_type='diag').to(self.device)
    max_ent_dist = torch.distributions.Uniform(s1.min(0).values, s1.max(0).values + 1e-6)
    self.model.mu.data = self.model.get_kmeans_mu(max_ent_dist.sample((1000, )).to(self.device), self.n_components)
    self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
    self.parent_set_idxs = parent_set_idxs

  def fit_parent_data(self, parent_data, epochs, use_tqdm=True, plot_callback=lambda _: None):
    """
    Global fit to parent data joint.
    """

    res = []
    dataloader = DataLoader(TensorDataset(torch.Tensor(parent_data)),
                            batch_size=self.batch_size,
                            shuffle=True,
                            num_workers=self.n_workers,
                            drop_last=True)

    epochiter = tqdm.tqdm(range(epochs)) if use_tqdm else range(epochs)
    for _ in epochiter:
      for batch in dataloader:
        s1 = batch[0].to(self.device)
        loss = -self.model.score_samples(s1).sum()
        res.append(loss.item())
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

      plot_callback(self)
    return res

  def generate_parents(self, n_samples):
    """
    Should generate a numpy array of shape (n_samples, parent_dims). 
    """
    return self.model.sample(n_samples)

class DisentangledGMMParentsModel(GlobalGMMParentsModel):
  
  def name(self):
    return 'DisentangledGMM'
  
  def fit_parent_data(self, parent_data, epochs, max_ent=True, use_tqdm=True, plot_callback=lambda _: None):
    """
    Should fit the parent_data to match the marginals/support of each parent_set,
      while maximizing entropy/matching a target distribution.
    """

    res = []
    dataloader = DataLoader(TensorDataset(torch.Tensor(parent_data)),
                            batch_size=self.batch_size,
                            shuffle=True,
                            num_workers=self.n_workers,
                            drop_last=True)

    epochiter = tqdm.tqdm(range(epochs)) if use_tqdm else range(epochs)
    for _ in epochiter:
      for batch in dataloader:
        s1 = batch[0].to(self.device)
        loss = torch.tensor(0.).to(self.device)

        # fit marginals
        for idxs in self.parent_set_idxs:
          loss -= self.model.score_samples(s1[:, idxs], idxs=idxs).sum()
        
        if max_ent:
          # max entropy
          entropies = diag_gaussian_batch_entropies(self.model.var.square()[0])
          c = torch.softmax(self.model.logpi, dim=1)[0, :, 0]
          conditional_entropy = conditional_entropy_gmm(c, entropies)
          D = pairwise_chernoff_divergences(0.5, self.model.mu[0], self.model.var[0].square())

          loss += 0.2*self.model.score_samples(self.model.sample(256).detach()).mean()

        # optimize
        res.append(loss.item())
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

      plot_callback(self)
    return res

class ConditionalGMMParentsModel(ParentsModel):

  def name(self):
    return 'ConditionalGMM'

  def __init__(self, n_components=32, device='cuda'):
    self.n_components = n_components
    self.models = []
    self.device = device
    self.parent_sample = None

  def setup_parent_models(self, parent_data, parent_set_idxs):
    s1 = torch.Tensor(parent_data)[:100000].to(self.device)
    self.parent_sample = s1
    self.parent_set_idxs = parent_set_idxs

    for idxs in parent_set_idxs:
      self.models.append(GMM(self.n_components, len(idxs), covariance_type='diag').to(self.device))

  def fit_parent_data(self, parent_data):
    """
    Fits each model to its marginal.
    """
    data = torch.Tensor(parent_data)

    for idxs, model in zip(self.parent_set_idxs, self.models):
      print('Fitting', idxs)
      tmp_data = data[:,idxs].to(self.device)
      model.fit(tmp_data)
      del tmp_data
      gc.collect()

  def generate_parents(self, n_samples):
    """
    Should generate a numpy array of shape (n_samples, parent_dims). 
    """
    n = max(chain.from_iterable(self.parent_set_idxs))+1
    SAMPLED_IDXS = set()
    res = np.zeros((n_samples, n))

    g = np.random.permutation(np.arange(len(self.models)))


    for idxs, model in zip(np.array(self.parent_set_idxs, dtype='object')[(g,)], np.array(self.models, dtype='object')[(g,)]):
      gmm = StaticGMM(len(idxs), self.n_components, mus=model.mu.data[0].cpu().numpy(), stds=model.var.data[0].cpu().numpy())
      already_sampled = [(i in SAMPLED_IDXS) for i in idxs]
      not_already_sampled = [not (i in SAMPLED_IDXS) for i in idxs]

      if np.any(already_sampled):
        cgmm = condition_gmm(gmm, np.arange(len(idxs))[already_sampled])
        res[:,idxs[not_already_sampled]] = cgmm.sample(res[:,idxs[already_sampled]])
      else:
        res[:,idxs.astype(np.int32)] = gmm.sample(len(res))
    return res

class ConditionalGMMDynamicsModel(DynamicsModel):

  def __init__(self, parents_model, n_components = 8, device = 'cuda'):
    super().__init__(parents_model)
    self.n_components = n_components
    self.device = device
    self.parent_idxs = []
    self.child_idxs = []
    self.models = []

  def name(self):
    return 'ConditionalGMM'

  def setup_dynamics_models(self, flat_sas_data, roots, children):
    """
    Should set up the dynamics models.
    """

    for child in children.values():
      self.parent_idxs.append(sum([roots[p]['idxs'] for p in child['parents']], []))
      self.child_idxs.append(child['idxs'])
      self.models.append(GMM(self.n_components, len(self.parent_idxs[-1]) + len(self.child_idxs[-1]), covariance_type='diag').to(self.device))

  def fit_dynamics(self, flat_sas_data):
    """
    Should fit the dynamics of the full dataset using a disentangled dynamics model.
    """
    data = torch.Tensor(flat_sas_data)

    for pidxs, cidxs, model in zip(self.parent_idxs, self.child_idxs, self.models):
      print('Fitting', pidxs + cidxs)
      tmp_data = data[:,pidxs + cidxs].to(self.device)
      model.fit(tmp_data)
      del tmp_data
      gc.collect()

  def generate_children(self, parents):
    """
    Should generate a numpy array of shape (len(parents), child_dims). 
    """
    all_child_idxs = sum(self.child_idxs, [])
    min_child_idx = min(all_child_idxs)
    res = np.zeros((len(parents), len(all_child_idxs)))

    g = np.random.permutation(np.arange(len(self.models)))

    for pidxs, cidxs, model in zip(self.parent_idxs, self.child_idxs, self.models):
      gmm = StaticGMM(len(pidxs) + len(cidxs), self.n_components, mus=model.mu.data[0].cpu().numpy(), stds=model.var.data[0].cpu().numpy())

      cgmm = condition_gmm(gmm, list(range(len(pidxs))))
      res[:,np.array(cidxs) - min_child_idx] = cgmm.sample(parents[:, pidxs])

    return res

class SimpleMLP(nn.Module):
  """Standard feedforward NN with Relu activations.
  Layers should include both input and output dimensions."""
  def __init__(self, layers):
    super().__init__()
    layer_list = []
    for i, o in zip(layers[:-1], layers[1:]):
      layer_list += [nn.Linear(i, o), nn.ReLU()]
    layer_list = layer_list[:-1]
    self.f = nn.Sequential(*layer_list)

  def forward(self, x):
    return self.f(x)

class EnsembleDynamicsModel(nn.Module):
  def __init__(self, model_fn, ensemble_size, lr=1e-4, device='cuda'):
    super().__init__()
    self.ensemble_size = ensemble_size
    self.device = device
    self.models = nn.ModuleList([model_fn().to(self.device) for _ in range(self.ensemble_size)])
    params = sum([list(model.parameters()) for model in self.models], [])
    self.params = params
    self.optimizer = torch.optim.Adam(params, lr=lr, weight_decay=5e-5)

  def forward(self, sa, concentration=1.):
    """1-step forward prediction; operates on raw states/actions"""

    res = torch.stack([m(sa) for m in self.models]) # ensemble_size x batch x 2*output

    preds, log_sigmasqs = torch.chunk(res, 2, 2)
    sigmasqs = torch.nn.functional.softplus(log_sigmasqs) + 1e-6
    sigmas = torch.sqrt(sigmasqs) / concentration
    next_state_dists = torch.distributions.Normal(preds, sigmas)
    next_states = next_state_dists.rsample() # ensemble_size x batch x output
    next_states = [ns.cpu().numpy() for ns in torch.chunk(next_states, self.ensemble_size, 0)]
    next_states = np.concatenate(next_states).transpose((1, 0, 2))
    next_states = next_states[np.arange(len(next_states)), np.random.randint(self.ensemble_size, size=len(next_states))]
    return next_states

  def evaluate(self, sa, next_states):
    res = torch.stack([m(sa) for m in self.models]) # ensemble_size x batch x 2*output
    preds, log_sigmasqs = torch.chunk(res, 2, 2)
    sigmasqs = torch.nn.functional.softplus(log_sigmasqs) + 1e-6

    loss = ((preds - next_states[None])**2 / (2 * sigmasqs) + 0.5 * torch.log(sigmasqs)).mean()

    return loss
    

  def train_on_sa_s_batch(self, sa, next_states):
    loss = self.evaluate(sa, next_states)

    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    return float(loss.item())

class MLPDynamicsModel(DynamicsModel):

  def __init__(self, parents_model, batch_size = 512, hidden_layers=[200]*4, ensemble_size=5, n_workers = 4, lr=1e-4, device = 'cuda'):
    super().__init__(parents_model)
    self.device = device

    self.hidden_layers = hidden_layers
    self.batch_size = batch_size
    self.optimizer = None
    self.ensemble_size = ensemble_size
    self.n_workers = n_workers
    self.lr = lr

    self.parent_idxs = []
    self.child_idxs = []
    self.models = []

  def name(self):
    return self.parents_model.name() + '_MLPDyna'

  def setup_dynamics_models(self, _, roots, children):
    """
    Should set up the dynamics models.
    """

    for child in children.values():
      self.parent_idxs.append(sum([roots[p]['idxs'] for p in child['parents']], []))
      self.child_idxs.append(child['idxs'])
      self.models.append(
        EnsembleDynamicsModel(
          model_fn=lambda: SimpleMLP(
            layers=[len(self.parent_idxs[-1])] + self.hidden_layers + [2*len(self.child_idxs[-1])]
          ),
          ensemble_size=self.ensemble_size,
          lr=self.lr,
          device=self.device
        )
      )

  def fit_dynamics(self, flat_sas_data, epochs, num_validation_samples=20000, validate_every=100, use_tqdm=True, plot_callback=lambda _: None):
    """
    Should fit the dynamics of the full dataset using a disentangled dynamics model.
    """
    tr_res = []
    val_res = []

    flat_sas_data = flat_sas_data[np.random.permutation(len(flat_sas_data))]

    tr_data = flat_sas_data[:-num_validation_samples]
    validation_data = torch.Tensor(flat_sas_data[-num_validation_samples:]).to(self.device)

    epochiter = tqdm.tqdm(range(epochs)) if use_tqdm else range(epochs)
    j = 0

    N = len(self.models)
    for epoch in epochiter:
      data = tr_data[np.random.permutation(len(tr_data))[:40000]]
      dataloader = DataLoader(TensorDataset(torch.Tensor(data)),
                              batch_size=self.batch_size,
                              shuffle=True,
                              num_workers=self.n_workers,
                              drop_last=True)
      for _, batch in enumerate(dataloader):
        s1 = batch[0].to(self.device)
        loss = 0
  
        for pidxs, cidxs, model in zip(self.parent_idxs, self.child_idxs, self.models):
          loss += (1./N) * model.train_on_sa_s_batch(s1[:,pidxs], s1[:,cidxs])
        
        if j % validate_every == 0:
          tr_res.append(loss)
          with torch.no_grad():
            loss = 0
      
            for pidxs, cidxs, model in zip(self.parent_idxs, self.child_idxs, self.models):
              loss += (1./N) * model.evaluate(validation_data[:,pidxs], validation_data[:,cidxs])

          val_res.append(loss.item())
          if epochs - epoch < 50:
            if val_res[-1] < min(val_res[-10:-1]):
              break
        j += 1
      if epochs - epoch < 50:
        if val_res[-1] < min(val_res[-10:-1]):
          break

      plot_callback(self)
    return tr_res, val_res

  def generate_children(self, parents):
    """
    Should generate a numpy array of shape (len(parents), child_dims). 
    """
    parents = torch.tensor(parents, dtype=torch.float32).to(self.device)
    all_child_idxs = sum(self.child_idxs, [])
    min_child_idx = min(all_child_idxs)
    res = np.zeros((len(parents), len(all_child_idxs)))

    with torch.no_grad():
      for pidxs, cidxs, model in zip(self.parent_idxs, self.child_idxs, self.models):
        res[:,np.array(cidxs) - min_child_idx] = model.forward(parents[:,pidxs], concentration=3.)

    return res

class MaskedComposer(nn.Module):
  def __init__(self, splits, out_dim, activation=nn.Identity()):
    """Splits is a list of indice lists, one for each parent, e.g., [[0,1], [2]]"""
    super().__init__()
    self.Gammas = nn.ModuleList([nn.Linear(len(s), out_dim) for s in splits])
    self.splits = splits
    self.out_dim = out_dim
    self.activation = activation

  def forward(self, x, mask):
    # x should be |batch_dim| x |total_in|
    # mask, if provided, should be binary B x num_splits

    res = torch.zeros(x.shape[0], self.out_dim, device=x.device)
    for i, s in enumerate(self.splits):
      res += self.activation(self.Gammas[i](x[:,s])) * mask[:,i,None]
      
    return res

class MaskedMLPDynamicsModel(DynamicsModel):

  def __init__(self, parents_model, batch_size = 512, hidden_layers=[200]*3, ensemble_size=5, n_workers = 4, lr=1e-4, device = 'cuda'):
    super().__init__(parents_model)
    self.device = device

    self.hidden_layers = hidden_layers
    self.batch_size = batch_size
    self.optimizer = None
    self.ensemble_size = ensemble_size
    self.n_workers = n_workers
    self.lr = lr

    self.parent_idxs = []
    self.all_parents = []
    self.child_idxs = []
    self.models = []
    self.opts = []

  def name(self):
    return self.parents_model.name() + '_MaskedMLPDyna'

  def setup_dynamics_models(self, _, roots, children, mask_fn):
    """
    Should set up the dynamics models.
    """
    self.mask_fn = mask_fn # accepts B x total_in_dim and produces B x num_roots x num_children
    self.parent_idxs = [r['idxs'] for r in roots.values()]
    self.all_parents = sum(self.parent_idxs, [])


    for child in children.values():
      self.child_idxs.append(child['idxs'])
      self.models.append(
        nn.ModuleList([
          MaskedComposer(self.parent_idxs, 128).to(self.device),
          EnsembleDynamicsModel(
            model_fn=lambda: SimpleMLP(
              layers=[128] + self.hidden_layers + [2*len(self.child_idxs[-1])]
            ),
            ensemble_size=self.ensemble_size,
            device=self.device
          )
        ])
      )
      self.opts.append(
        torch.optim.Adam(list(self.models[-1][0].parameters()) + list(self.models[-1][1].parameters()), lr=self.lr, weight_decay=5e-5)
      )

  def fit_dynamics(self, flat_sas_data, epochs, num_validation_samples=20000, validate_every=100, use_tqdm=True, plot_callback=lambda _: None):
    """
    Should fit the dynamics of the full dataset using the masked dynamics model.
    """ 
    tr_res = []
    val_res = []

    flat_sas_data = flat_sas_data[np.random.permutation(len(flat_sas_data))]

    tr_data = flat_sas_data[:-num_validation_samples]

    validation_data = torch.Tensor(flat_sas_data[-num_validation_samples:]).to(self.device)
    validation_mask = self.mask_fn(validation_data[:,self.all_parents])


    epochiter = tqdm.tqdm(range(epochs)) if use_tqdm else range(epochs)
    N = len(self.models)
    j = 0
    for epoch in epochiter:
      data = tr_data[np.random.permutation(len(tr_data))[:40000]]
      dataloader = DataLoader(TensorDataset(torch.Tensor(data)),
                              batch_size=self.batch_size,
                              shuffle=True,
                              num_workers=self.n_workers,
                              drop_last=True)

      for batch in dataloader:
        s1 = batch[0].to(self.device)
        loss = 0

        mask = self.mask_fn(s1[:,self.all_parents])
  
        for i, (cidxs, model, opt) in enumerate(zip(self.child_idxs, self.models, self.opts)):
          composer, fwd_ensemble = model
          
          embeds = composer.forward(s1[:,self.all_parents], mask[:,:,i]) # should be batch_size x 256 [fixed default emb size]
          _loss = fwd_ensemble.evaluate(embeds, s1[:,cidxs])

          opt.zero_grad()
          _loss.backward()
          opt.step()

          loss += (1./N) * float(_loss.item())
        
        if j % validate_every == 0:
          tr_res.append(loss)
          with torch.no_grad():
            loss = 0
      
            for i, (cidxs, model) in enumerate(zip(self.child_idxs, self.models)):
              composer, fwd_ensemble = model
          
              embeds = composer.forward(validation_data[:,self.all_parents], validation_mask[:,:,i]) # should be batch_size x 256 [fixed default emb size]
              loss += (1./N) * fwd_ensemble.evaluate(embeds, validation_data[:,cidxs])

          val_res.append(loss.item())
          if epochs - epoch < 50:
            if val_res[-1] < min(val_res[-10:-1]):
              break
        j += 1
      if epochs - epoch < 50:
        if val_res[-1] < min(val_res[-10:-1]):
          break

      plot_callback(self)
    return tr_res, val_res

  def generate_children(self, parents):
    """
    Should generate a numpy array of shape (len(parents), child_dims). 
    """
    parents = torch.tensor(parents, dtype=torch.float32).to(self.device)
    mask = self.mask_fn(parents)

    all_child_idxs = sum(self.child_idxs, [])
    min_child_idx = min(all_child_idxs)
    res = np.zeros((len(parents), len(all_child_idxs)))

    with torch.no_grad():
      for i, (cidxs, model) in enumerate(zip(self.child_idxs, self.models)):
        composer, fwd_ensemble = model
    
        embeds = composer.forward(parents, mask[:,:,i]) # should be batch_size x 256 [fixed default emb size]
        res[:,np.array(cidxs) - min_child_idx] = fwd_ensemble.forward(embeds, concentration=3.)

    return res
