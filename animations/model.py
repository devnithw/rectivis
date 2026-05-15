# -*- coding: utf-8 -*-
"""
Neural network models and ODE solvers.
"""

import torch
import torch.nn as nn
from animations.config import MODEL_CONFIG, TRAINING_CONFIG


class MLP(nn.Module):
    """Simple 3-layer MLP for velocity field prediction."""
    
    def __init__(self, input_dim=None, hidden_num=None):
        super().__init__()
        if input_dim is None:
            input_dim = MODEL_CONFIG["input_dim"]
        if hidden_num is None:
            hidden_num = MODEL_CONFIG["hidden_dim"]
        
        self.fc1 = nn.Linear(input_dim + 1, hidden_num, bias=True)
        self.fc2 = nn.Linear(hidden_num, hidden_num, bias=True)
        self.fc3 = nn.Linear(hidden_num, input_dim, bias=True)
        self.act = lambda x: torch.tanh(x)

    def forward(self, x_input, t):
        """
        Forward pass.
        
        Args:
            x_input: Position tensor of shape (batch_size, input_dim)
            t: Time tensor of shape (batch_size, 1)
        
        Returns:
            velocity: Velocity prediction of shape (batch_size, input_dim)
        """
        inputs = torch.cat([x_input, t], dim=1)
        x = self.fc1(inputs)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        return x


class FlowMatching:
    """Flow Matching model using straight-line interpolation."""
    
    def __init__(self, model=None, num_steps=None):
        self.model = model
        self.num_steps = num_steps or TRAINING_CONFIG["num_steps"]

    def get_train_tuple(self, z0=None, z1=None):
        """
        Generate training data: (z_t, t, velocity_target).
        
        Args:
            z0: Initial samples
            z1: Target samples
        
        Returns:
            z_t: Interpolated positions
            t: Time values
            target_velocity: Target velocity (z1 - z0)
        """
        t = torch.rand((z1.shape[0], 1), device=z1.device)
        z_t = (1.0 - t) * z0 + t * z1
        target_velocity = z1 - z0
        return z_t, t, target_velocity

    @torch.no_grad()
    def sample_ode(self, z0=None, num_steps=None):
        """
        Solve ODE using Euler method.
        
        Args:
            z0: Initial positions
            num_steps: Number of integration steps
        
        Returns:
            trajectory: List of position tensors over time
        """
        if num_steps is None:
            num_steps = self.num_steps
        
        dt = 1.0 / num_steps
        trajectory = [z0.detach().clone()]
        z = z0.detach().clone()
        batchsize = z.shape[0]

        for i in range(num_steps):
            t = torch.ones((batchsize, 1), device=z.device) * i / num_steps
            velocity = self.model(z, t)
            z = z.detach().clone() + velocity * dt
            trajectory.append(z.detach().clone())

        return trajectory


class RectifiedFlow:
    """Rectified Flow model with ODE sampling."""
    
    def __init__(self, model=None, num_steps=None):
        self.model = model
        self.num_steps = num_steps or TRAINING_CONFIG["num_steps"]

    def get_train_tuple(self, z0=None, z1=None):
        """
        Generate training data: (z_t, t, velocity_target).
        
        Args:
            z0: Initial samples
            z1: Target samples
        
        Returns:
            z_t: Interpolated positions
            t: Time values
            target_velocity: Target velocity (z1 - z0)
        """
        t = torch.rand((z1.shape[0], 1), device=z1.device)
        z_t = t * z1 + (1.0 - t) * z0
        target_velocity = z1 - z0
        return z_t, t, target_velocity

    @torch.no_grad()
    def sample_ode(self, z0=None, num_steps=None):
        """
        Solve ODE using Euler method.
        
        Args:
            z0: Initial positions
            num_steps: Number of integration steps
        
        Returns:
            trajectory: List of position tensors over time
        """
        if num_steps is None:
            num_steps = self.num_steps
        
        dt = 1.0 / num_steps
        trajectory = [z0.detach().clone()]
        z = z0.detach().clone()
        batchsize = z.shape[0]

        for i in range(num_steps):
            t = torch.ones((batchsize, 1), device=z.device) * i / num_steps
            velocity = self.model(z, t)
            z = z.detach().clone() + velocity * dt
            trajectory.append(z.detach().clone())

        return trajectory


class ODEModel:
    """Generic ODE solver wrapper for animation generation."""
    
    def __init__(self, model, num_steps=100):
        self.model = model
        self.num_steps = num_steps

    @torch.no_grad()
    def get_trajectory(self, z0, num_steps=None):
        """
        Get full trajectory for batch of particles.
        
        Args:
            z0: Initial positions
            num_steps: Number of integration steps
        
        Returns:
            trajectory: Tensor of shape (num_steps+1, batch_size, 2)
        """
        if num_steps is None:
            num_steps = self.num_steps
        
        dt = 1.0 / num_steps
        trajectory = [z0.detach().clone()]
        z = z0.detach().clone()
        batchsize = z.shape[0]

        for i in range(num_steps):
            t = torch.ones((batchsize, 1), device=z.device) * i / num_steps
            velocity = self.model(z, t)
            z = z.detach().clone() + velocity * dt
            trajectory.append(z.detach().clone())

        return torch.stack(trajectory, dim=0)  # Shape: (num_steps+1, batch, 2)

