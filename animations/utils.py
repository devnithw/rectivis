# -*- coding: utf-8 -*-
"""
Utility functions for training and model management.
"""

import torch
import numpy as np
import random
from animations.config import TRAINING_CONFIG


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def train_model(model_wrapper, optimizer, pairs, batch_size, num_iterations, 
               verbose=True, verbose_interval=2000):
    """
    Generic training function for flow models.
    
    Args:
        model_wrapper: FlowMatching or RectifiedFlow instance
        optimizer: PyTorch optimizer
        pairs: Data pairs tensor of shape (n_pairs, 2, 2)
        batch_size: Batch size
        num_iterations: Number of iterations
        verbose: Print progress
        verbose_interval: Interval for printing
    
    Returns:
        model_wrapper: Updated model wrapper
        loss_curve: List of log loss values
    """
    loss_curve = []
    
    for i in range(num_iterations + 1):
        optimizer.zero_grad()
        
        indices = torch.randperm(len(pairs))[:batch_size]
        batch = pairs[indices]
        z0 = batch[:, 0].detach().clone()
        z1 = batch[:, 1].detach().clone()
        
        z_t, t, target = model_wrapper.get_train_tuple(z0=z0, z1=z1)
        
        pred = model_wrapper.model(z_t, t)
        loss = (target - pred).view(pred.shape[0], -1).pow(2).sum(dim=1).mean()
        
        loss.backward()
        optimizer.step()
        
        loss_curve.append(np.log(loss.item()))
        
        if verbose and (i + 1) % verbose_interval == 0:
            print(f"Iteration {i+1}/{num_iterations}, Loss: {loss.item():.6f}")
    
    return model_wrapper, loss_curve


def save_model(model, filepath):
    """Save model state dict."""
    torch.save(model.state_dict(), filepath)
    print(f"Saved model: {filepath}")


def load_model(model_class, filepath, device='cpu', **kwargs):
    """
    Load model from checkpoint.
    
    Args:
        model_class: Model class to instantiate
        filepath: Path to saved state dict
        device: Device to load to
        **kwargs: Additional arguments for model initialization
    
    Returns:
        model: Loaded model
    """
    model = model_class(**kwargs)
    model.load_state_dict(torch.load(filepath, map_location=device))
    model.to(device)
    model.eval()
    return model


def copy_model(model):
    """Create a deep copy of model parameters."""
    import copy
    return copy.deepcopy(model)


def get_device():
    """Get appropriate device (GPU if available, else CPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def print_config(config_dict, title="Configuration"):
    """Pretty print configuration dictionary."""
    print("\n" + "="*60)
    print(f"{title}")
    print("="*60)
    for key, value in config_dict.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")
    print("="*60 + "\n")
