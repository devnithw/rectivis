import torch
import numpy as np
import torch.nn as nn
from torch.distributions import Normal, Categorical
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.mixture_same_family import MixtureSameFamily

class UnimodalDistribution:    
    def __init__(self, mean, variance):
        self.mean = torch.tensor(mean).float()
        self.variance = torch.tensor(variance).float()
        self.model = Normal(self.mean, torch.sqrt(self.variance))
    
    def sample(self, n_samples):
        return self.model.sample([n_samples])

class BimodalDistribution:
    def __init__(self, means, variance):
        self.means = means
        self.variance = variance
        self.model = MixtureSameFamily(
            Categorical(torch.tensor([0.5, 0.5])),
            MultivariateNormal(means, variance * torch.eye(2))
        )
    
    def sample(self, n_samples):
        return self.model.sample([n_samples])

class NModalDistribution:
    pass

class CheckerboardDistribution:
    pass
