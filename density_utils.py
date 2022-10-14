import torch
import torch.nn as nn
import torch.distributions as D
import numpy as np
from scipy.stats import norm
from sklearn.neighbors import KernelDensity

class StaticGMM():
  """
  TRUNCATED SAMPLING TO 2.5 STDS
  """
  def __init__(self, n, num_components, mus=None, stds=None, ps=None):
    if mus is not None:
      assert mus.shape == (num_components, n)
      self.mus = mus
    else:
      self.mus = 5 * np.random.randn(num_components, n)
      
    self.stds = stds if stds is not None else (np.random.random((num_components, n)) + 0.5)
    self.ps = ps if ps is not None else np.ones((num_components,), np.float32) / num_components
    
    self.n = n
    self.num_components = num_components

  def sample(self, m):
    """Samples m points"""
    cs = np.random.choice(self.num_components, size = m, p = self.ps)
    one_hot_cs = np.eye(self.num_components)[cs][:,:,None]
    mus = (self.mus[None] * one_hot_cs).sum(1) # m x n
    stds = (self.stds[None] * one_hot_cs).sum(1) # m x n

    samples = np.clip(np.random.randn(m, self.n), -2.5, 2.5) * stds + mus
    
    return samples.astype(np.float32)
  
  def density(self, points):
    """Computes unnormalized pdf of points (shape m x n)"""
    points = np.tile(points[:,None,:], [1, self.num_components, 1])
    normed = (points - self.mus[None]) / self.stds[None]
    pdfs = norm.pdf(normed) # m x num_components x n
    return  (pdfs.prod(-1) * self.ps[None]).sum(1)

def compute_kernel(x, y, kernel='gaussian', sigma=1.):
  """Inputs will be (batch_size / sample_size, sample_size, features)."""
  x = x[:,:,None,:] # (B, x_size, 1, dim)
  y = y[:,None] # (B, 1, y_size, dim)
  if kernel=='gaussian':
    kernel_input = (x - y).pow(2).sum(-1)/(2. * sigma)
  elif kernel=='laplace':
    kernel_input = (x - y).abs().sum(-1)/(2. * sigma)
  else:
    raise NotImplemented
  return torch.exp(-kernel_input) # (x_size, y_size)

def compute_mmd(x, y, n_samples=None):
  """Input will be (batch_size, features). If n_samples is not none,
  then n_samples must divide batch_size, and MMD will be computed in
  subbatches of size n_samples."""
  assert len(x) == len(y)
  batch_size = len(x)
  if n_samples is None:
    n_samples = batch_size
  assert batch_size % n_samples == 0
  
  x = x.reshape(batch_size // n_samples, n_samples, -1)
  y = y.reshape(batch_size // n_samples, n_samples, -1)
  
  x_kernel = compute_kernel(x, x).mean(dim=(1,2)) # (B // S,)
  y_kernel = compute_kernel(y, y).mean(dim=(1,2)) # (B // S,)
  xy_kernel = compute_kernel(x, y).mean(dim=(1,2)) # (B // S,)
  
  mmds = (x_kernel + y_kernel - 2*xy_kernel + 1e-6).sqrt()
  
  return mmds.mean()

class MLP(nn.Module):
  """Standard feedforward network.
  Args:
    input_size (int): number of input features
    output_size (int): number of output features
    hidden_sizes (int tuple): sizes of hidden layers
    activ: activation module (e.g., GELU, nn.ReLU)
    drop_prob: dropout probability to apply between layers (not applied to input)
  """
  def __init__(self, input_size, output_size=1, hidden_sizes=(256, 256), activ=nn.ReLU, drop_prob=0.):
    super().__init__()
    self.output_size = output_size

    layer_sizes = (input_size, ) + tuple(hidden_sizes) + (output_size, )
    if len(layer_sizes) == 2:
      layers = [nn.Linear(layer_sizes[0], layer_sizes[1], bias=False)]
    else:
      layers = []
      for dim_in, dim_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        layers.append(activ())
        if drop_prob > 0.:
          layers.append(nn.Dropout(p=drop_prob))
      layers = layers[:-(1 + (drop_prob > 0))]
    self.f = nn.Sequential(*layers)

  def forward(self, x):
    return self.f(x)
  
def build_generator(noise_dim=8, hidden_dim=256, output_dim=22, device='cuda'):
  noise = D.Normal(torch.zeros(noise_dim).to(device), torch.ones(noise_dim).to(device))
  mlp = MLP(noise_dim, output_dim, (hidden_dim, hidden_dim)).to(device)
  opt = torch.optim.Adam(params=mlp.parameters(), lr=1e-3, weight_decay=1e-4)
  def generate(n):
    return mlp(noise.sample((n,)).cuda())
  return generate, opt

def scatter2Dsamples(s1, s2, plt, new_figure=True):
  if new_figure:
    plt.figure()
  if type(s1) == torch.Tensor:
    s1 = s1.detach().cpu()
  if type(s2) == torch.Tensor:
    s2 = s2.detach().cpu()
  plt.scatter(s1[:,0],s1[:,1], s=8, alpha=0.3, c = 'b')
  plt.scatter(s2[:,0],s2[:,1], s=8, alpha=0.3, c = 'r')


class FeaturewiseKDE():
  def __init__(self, data, max_samples=5000, kernel='gaussian', bandwidth=0.1, normalize=True):
    self.data = data.copy()
    self.normalize=normalize
    
    self.data = self.data[np.random.permutation(len(self.data))[:max_samples]]
    self.mean = 0
    self.std = 1
    
    if self.normalize:
      self.mean = self.data.mean(0, keepdims=True)
      self.std = (self.data.std(0, keepdims=True) + 1e-4)
    
    self.data -= self.mean
    self.data /= self.std
      
    self.kdes = []
    for feat in range(self.data.shape[-1]):
      kde = KernelDensity(kernel=kernel, bandwidth=bandwidth)
      kde.fit(self.data[:,feat:feat+1])
      self.kdes.append(kde)      

  def nonOutlierIdxs(self, data, m = 5):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return (s<m)
      
  def clean_outliers(self, sample, m = 5):
    
    sample = sample.copy()
    
    sample -= self.mean
    sample /= self.std
    
    for feat, kde in enumerate(self.kdes):
      scores = kde.score_samples(sample[:,feat:feat+1])
      safe = self.nonOutlierIdxs(scores, m=m)
      sample = sample[safe]
    
    sample *= self.std
    sample += self.mean
    
    return sample