# -*- coding: utf-8 -*-
"""
Data generation utilities for distributions.
"""

import torch
from torch.distributions import MultivariateNormal, Categorical
from torch.distributions.mixture_same_family import MixtureSameFamily
from animations.config import DISTRIBUTION_CONFIG


def create_unimodal_to_bimodal(device='cpu'):
    """
    Scenario 1: Unimodal Gaussian → Bimodal Gaussian
    
    Initial: Unimodal Gaussian at (-D, 0)
    Target: Bimodal Gaussian with modes at (D, ±D/2)
    
    Args:
        device: torch device (cpu or cuda)
    
    Returns:
        samples_0: Samples from initial unimodal Gaussian
        samples_1: Samples from target bimodal Gaussian
    """
    D = DISTRIBUTION_CONFIG["d"]
    var_init = DISTRIBUTION_CONFIG["initial_var"]
    var_target = DISTRIBUTION_CONFIG["target_var"]
    n_samples = DISTRIBUTION_CONFIG["n_samples"]
    n_per_mode = DISTRIBUTION_CONFIG["n_samples_per_mode"]
    
    # Initial: Unimodal Gaussian centered at (-D, 0)
    initial_mean = torch.tensor([-D, 0.0], dtype=torch.float32)
    initial_cov = torch.eye(2) * var_init
    initial_dist = MultivariateNormal(initial_mean, initial_cov)
    samples_0 = initial_dist.sample([n_samples]).to(device)

    # Target: Bimodal Gaussian (two modes at (D, D/2) and (D, -D/2))
    mode1_mean = torch.tensor([D, D / 2.0], dtype=torch.float32)
    mode2_mean = torch.tensor([D, -D / 2.0], dtype=torch.float32)
    cov = torch.eye(2) * var_target

    samples_mode1 = MultivariateNormal(mode1_mean, cov).sample([n_per_mode]).to(device)
    samples_mode2 = MultivariateNormal(mode2_mean, cov).sample([n_per_mode]).to(device)
    samples_1 = torch.cat([samples_mode1, samples_mode2], dim=0).to(device)
    samples_1 = samples_1[torch.randperm(len(samples_1))]

    return samples_0, samples_1


def create_bimodal_to_bimodal(device='cpu'):
    """
    Scenario 2: Bimodal Gaussian → Bimodal Gaussian (translated)
    
    Initial: Bimodal Gaussian with modes at (-D, ±sep/2)
    Target: Bimodal Gaussian with modes at (D, ±sep/2)
    Both distributions have same structure, translated along x-axis.
    
    Args:
        device: torch device (cpu or cuda)
    
    Returns:
        samples_0: Samples from initial bimodal Gaussian
        samples_1: Samples from target bimodal Gaussian
    """
    D = DISTRIBUTION_CONFIG["d"]
    var = DISTRIBUTION_CONFIG["target_var"]
    n_per_mode = DISTRIBUTION_CONFIG["n_samples_per_mode"]
    sep = DISTRIBUTION_CONFIG["bimodal_separation"]
    
    # Initial: Bimodal Gaussian with modes at (-D, ±sep/2)
    mode1_init = torch.tensor([-D, sep / 2.0], dtype=torch.float32)
    mode2_init = torch.tensor([-D, -sep / 2.0], dtype=torch.float32)
    cov = torch.eye(2) * var
    
    samples_mode1_init = MultivariateNormal(mode1_init, cov).sample([n_per_mode]).to(device)
    samples_mode2_init = MultivariateNormal(mode2_init, cov).sample([n_per_mode]).to(device)
    samples_0 = torch.cat([samples_mode1_init, samples_mode2_init], dim=0).to(device)
    samples_0 = samples_0[torch.randperm(len(samples_0))]
    
    # Target: Bimodal Gaussian with modes at (D, ±sep/2)
    mode1_target = torch.tensor([D, sep / 2.0], dtype=torch.float32)
    mode2_target = torch.tensor([D, -sep / 2.0], dtype=torch.float32)
    
    samples_mode1_target = MultivariateNormal(mode1_target, cov).sample([n_per_mode]).to(device)
    samples_mode2_target = MultivariateNormal(mode2_target, cov).sample([n_per_mode]).to(device)
    samples_1 = torch.cat([samples_mode1_target, samples_mode2_target], dim=0).to(device)
    samples_1 = samples_1[torch.randperm(len(samples_1))]
    
    return samples_0, samples_1


def create_distributions(device='cpu'):
    """
    Create distributions based on selected scenario.
    
    Args:
        device: torch device (cpu or cuda)
    
    Returns:
        samples_0: Samples from initial distribution
        samples_1: Samples from target distribution
    """
    scenario = DISTRIBUTION_CONFIG["scenario"]
    
    if scenario == "unimodal_to_bimodal":
        return create_unimodal_to_bimodal(device)
    elif scenario == "bimodal_to_bimodal":
        return create_bimodal_to_bimodal(device)
    else:
        raise ValueError(f"Unknown scenario: {scenario}. Choose 'unimodal_to_bimodal' or 'bimodal_to_bimodal'.")


def create_data_pairs(samples_0, samples_1):
    """
    Create random pairings between initial and target samples.
    
    Args:
        samples_0: Initial distribution samples
        samples_1: Target distribution samples
    
    Returns:
        pairs: Stacked tensor of shape (n_samples, 2, 2)
    """
    x_0 = samples_0[torch.randperm(len(samples_0))]
    x_1 = samples_1[torch.randperm(len(samples_1))]
    pairs = torch.stack([x_0, x_1], dim=1)
    return pairs
