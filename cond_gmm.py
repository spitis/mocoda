from typing import Tuple
from typing import Union
from collections.abc import Iterable

import torch

from density_utils import StaticGMM
from truncated_normal_pytorch import TruncatedNormal

def marginalize_gmm(gmm, axis):
  return StaticGMM(
    n = len(axis),
    num_components=gmm.num_components,
    mus=gmm.mus[:,axis],
    stds=gmm.stds[:,axis],
    ps=gmm.ps
  )

def condition_gmm(
    gmm: StaticGMM,
    axis: Union[int, Tuple[int, ...]],
    ) -> object:
  # TODO(): better typing hints?
  """
  Starting with a joint GMM, condition on specified axes.

  NOTE: This uses a truncated normal, truncated to +-2.5 STDs.
  """

  class CondStaticGMM():
    """Class similar to StaticGMM, but every function requires conditioning."""
    def __init__(self, gmm, axis):
      self.joint_gmm = gmm
      self.axis = axis if isinstance(axis, Iterable) else [axis]
      self.not_axis = [i for i in range(gmm.n) if i not in self.axis]

    def conditional_gmm(self, x):
      """Produce a conditional GMM for dims not in self.axis, conditioned on x.
      
      i.e. given joint GMM p(x, y) and graph y <= z => x, compute a GMM p(y|x),
      which works out to E_{p(z|x)}[ p(y|z) ]. The key insight are 
        1) from the GMM p(x,y|z)p(z), marginal GMMs p(x|z)p(z) and p(y|z)p(z)
           are readily available. B/c component Gaussians are diagonal, this is 
           especially easy.
        2) p(z|x) are proportional to the prior probs p(z) times p(x|z),
           evaluated on the observed samples.
      """
      # for book keeping: store dist'ns and log probs of samples
      P_utils = dict() # store GMMs built with density_utils
      P_torch = dict() # store GMMs built with torch.distributions
      log_probs = dict() # store log probs of observed tensors
      #
      P_utils['x'] = marginalize_gmm(self.joint_gmm, self.axis) # p(x)
      P_utils['y'] = marginalize_gmm(self.joint_gmm, self.not_axis) # p(y)
      # re-build the GMM in torch for ease of handling batch data
      t = torch.tensor
      mix = torch.distributions.Categorical(t(P_utils['x'].ps))
      comp = torch.distributions.Independent(
          torch.distributions.Normal(
              t(P_utils['x'].mus), t(P_utils['x'].stds)
          ), 1)
      P_torch['x'] = torch.distributions.MixtureSameFamily(mix, comp) # p(x)
      # compute posteriors and update mixture weights for the conditional
      log_probs['x|z'] = P_torch['x'].component_distribution.log_prob(
          P_torch['x']._pad(t(x))
      ) # [batch x components]
      log_probs['z'] = P_torch['x'].mixture_distribution.probs.log() # [components]
      log_probs['x,z'] = log_probs['z'][None] + log_probs['x|z'] # [batch x components]
      log_probs['z|x'] = log_probs['x,z'] - torch.logsumexp(
          log_probs['x,z'], dim=-1, keepdim=True
      ) # [batch x components]
      #
      # build GMM for p(y|x) in torch
      # technically each x in the batch gets its own GMM, so we tile the params
      # for p(y|z), one per batch element...
      N = len(x)
      mus = t(P_utils['y'].mus).tile((N, 1, 1))
      stds = t(P_utils['y'].stds).tile((N, 1, 1))
      comp = torch.distributions.Independent(
          TruncatedNormal(mus, stds, -2.5, 2.5), 1
      )
      # ...and use mixture weights p(z|x) as discussed in the docstring
      # mix = torch.distributions.Categorical(logits=log_probs['z|x'])
      mix = torch.distributions.Categorical(log_probs['z|x'].exp())
      P_torch['y|x'] = torch.distributions.MixtureSameFamily(mix, comp) # p(y|x)

      return P_torch['y|x']

    def sample(self, x):
      """For each x in batch, compute conditional p(not x|x) and draw one sample."""
      return self.conditional_gmm(x).sample().detach().numpy()

    def density(self, points):
      """Computes normalized pdf of points (shape m x n)
      
      Note that the self.axis dims of points are conditioned on, and the rest
      are used to compute the conditional density.
      """
      N, d = points.shape
      # axis = self.axis if isinstance(self.axis, Iterable) else [self.axis]
      # not_axis = [i for i in range(d) if i not in axis]
      x = points[:, self.axis] # condition on these
      y = points[:, self.not_axis] # compute density for these conditioned on x
      return self.conditional_gmm(x).log_prob(y).exp().detach().numpy()
  
    def compute_kernel(x, y, kernel='gaussian', sigma=1.):
      """Inputs will be (batch_size / sample_size, sample_size, features)."""
      raise NotImplementedError # TODO(): implement if we decide it is needed
    
    def compute_mmd(x, y, n_samples=None):
      """Input will be (batch_size, features). If n_samples is not none,
      then n_samples must divide batch_size, and MMD will be computed in
      subbatches of size n_samples."""
      raise NotImplementedError # TODO(): implement if we decide it is needed

  return CondStaticGMM(gmm, axis)