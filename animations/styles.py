# -*- coding: utf-8 -*-
"""
Plotting utilities and styling functions.
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import torch
from animations.config import COLORS, PLOT_CONFIG, FONTS

# Configure matplotlib with CMU fonts and LaTeX
matplotlib.rcParams['font.serif'] = [FONTS["serif"]]
matplotlib.rcParams['font.sans-serif'] = [FONTS["sans_serif"]]
matplotlib.rcParams['figure.dpi'] = PLOT_CONFIG["dpi"]
matplotlib.rcParams['text.usetex'] = False  # Use mathtext instead of full LaTeX for compatibility


def setup_plot_style():
    """Setup global plot styling."""
    plt.rcParams['figure.dpi'] = PLOT_CONFIG["dpi"]
    plt.rcParams['font.serif'] = [FONTS["serif"]]
    plt.rcParams['font.sans-serif'] = [FONTS["sans_serif"]]
    plt.rcParams['text.usetex'] = False


def remove_axes(ax):
    """Remove tick marks and labels from axes."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])


def set_plot_limits(ax, xlim=None, ylim=None):
    """Set plot limits."""
    xlim = xlim or PLOT_CONFIG["xlim"]
    ylim = ylim or PLOT_CONFIG["ylim"]
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def configure_distribution_plot(ax):
    """Configure a distribution plot with standard settings."""
    set_plot_limits(ax)
    ax.set_aspect('equal')
    remove_axes(ax)
    ax.grid(False)


def plot_distribution_pair(ax, samples_initial, samples_target, 
                          title=None):
    """
    Plot initial and target distributions.
    
    Args:
        ax: Matplotlib axis
        samples_initial: Initial distribution samples
        samples_target: Target distribution samples
        title: Plot title
    """
    configure_distribution_plot(ax)
    
    ax.scatter(samples_initial[:, 0].cpu().numpy(), 
              samples_initial[:, 1].cpu().numpy(),
              alpha=PLOT_CONFIG["alpha_distribution"],
              s=PLOT_CONFIG["dot_size"],
              color=COLORS["initial"])
    
    ax.scatter(samples_target[:, 0].cpu().numpy(),
              samples_target[:, 1].cpu().numpy(),
              alpha=PLOT_CONFIG["alpha_distribution"],
              s=PLOT_CONFIG["dot_size"],
              color=COLORS["target"])
    
    if title:
        ax.set_title(title, fontsize=FONTS["title_size"], fontweight='bold')
    


def plot_initial_distributions(samples_initial, samples_target, 
                               title=None, filename=None):
    """
    Plot the initial and target distributions in a single figure.
    No generated points, trajectories, or sampling — just the raw distributions.
    
    Args:
        samples_initial: Samples from the initial distribution (pi_0)
        samples_target: Samples from the target distribution (pi_1)
        title: Optional plot title
        filename: Optional filename to save
    
    Returns:
        fig: Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=PLOT_CONFIG["figsize"])
    
    configure_distribution_plot(ax)
    
    ax.scatter(samples_initial[:, 0].cpu().numpy(), 
              samples_initial[:, 1].cpu().numpy(),
              alpha=PLOT_CONFIG["alpha_distribution"],
              s=PLOT_CONFIG["dot_size"],
              color=COLORS["initial"])
    
    ax.scatter(samples_target[:, 0].cpu().numpy(),
              samples_target[:, 1].cpu().numpy(),
              alpha=PLOT_CONFIG["alpha_distribution"],
              s=PLOT_CONFIG["dot_size"],
              color=COLORS["target"])
    
    if title:
        ax.set_title(title, fontsize=FONTS["title_size"], fontweight='bold')
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=PLOT_CONFIG["dpi"], bbox_inches='tight')
    
    return fig



def plot_trajectories(ax, trajectory_particles, samples_initial, samples_target,
                     num_trajectories=50, title=None):
    """
    Plot transport trajectories.
    
    Args:
        ax: Matplotlib axis
        trajectory_particles: Trajectory data of shape (time_steps, num_particles, 2)
        samples_initial: Initial distribution samples
        samples_target: Target distribution samples
        num_trajectories: Number of trajectories to plot
        title: Plot title
    """
    configure_distribution_plot(ax)
    
    # Plot distributions
    ax.scatter(samples_initial[:1000, 0].cpu().numpy(),
              samples_initial[:1000, 1].cpu().numpy(),
              alpha=PLOT_CONFIG["alpha_distribution"],
              s=PLOT_CONFIG["dot_size"],
              color=COLORS["initial"])
    
    ax.scatter(samples_target[:, 0].cpu().numpy(),
              samples_target[:, 1].cpu().numpy(),
              alpha=PLOT_CONFIG["alpha_distribution"],
              s=PLOT_CONFIG["dot_size"],
              color=COLORS["target"])
    
    # Plot trajectory lines
    for i in range(min(num_trajectories, trajectory_particles.shape[1])):
        ax.plot(trajectory_particles[:, i, 0].cpu().numpy(),
               trajectory_particles[:, i, 1].cpu().numpy(),
               alpha=PLOT_CONFIG["alpha_trajectory"],
               linewidth=PLOT_CONFIG["trajectory_linewidth"],
               color=COLORS["trajectory"])
    
    if title:
        ax.set_title(title, fontsize=FONTS["title_size"], fontweight='bold')


def plot_results_comparison(samples_initial, samples_target, trajectory,
                           title="Results Comparison"):
    """
    Create a 3-panel comparison plot: initial, final, trajectories.
    
    Args:
        samples_initial: Initial distribution samples
        samples_target: Target distribution samples
        trajectory: Full trajectory list
        title: Figure title
    
    Returns:
        fig: Matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=PLOT_CONFIG["figsize_wide"])
    fig.suptitle(title, fontsize=FONTS["title_size"], fontweight='bold')
    
    # Panel 1: Initial distribution
    plot_distribution_pair(axes[0], samples_initial, samples_initial,
                          title="Initial Distribution")
    axes[0].scatter(samples_initial[:, 0].cpu().numpy(),
                   samples_initial[:, 1].cpu().numpy(),
                   alpha=PLOT_CONFIG["alpha_distribution"],
                   s=PLOT_CONFIG["dot_size"],
                   color=COLORS["initial"])
    axes[0].scatter([], [], color=COLORS["target"], alpha=0)  # Dummy for consistency
    
    # Panel 2: Final comparison
    configure_distribution_plot(axes[1])
    axes[1].scatter(samples_target[:, 0].cpu().numpy(),
                   samples_target[:, 1].cpu().numpy(),
                   alpha=PLOT_CONFIG["alpha_distribution"],
                   s=PLOT_CONFIG["dot_size"],
                   color=COLORS["target"])
    axes[1].scatter(trajectory[-1][:, 0].cpu().numpy(),
                   trajectory[-1][:, 1].cpu().numpy(),
                   alpha=PLOT_CONFIG["alpha_generated"],
                   s=PLOT_CONFIG["dot_size"],
                   color=COLORS["generated"])
    axes[1].set_title("Final Distribution", fontsize=FONTS["title_size"], fontweight='bold')
    
    # Panel 3: Trajectories
    traj_particles = torch.stack(trajectory)
    plot_trajectories(axes[2], traj_particles, samples_initial, samples_target,
                     num_trajectories=50, title="Transport Trajectories")
    
    plt.tight_layout()
    return fig


def plot_loss_curve(loss_curve, iterations, title="Training Loss", filename=None):
    """
    Plot training loss curve.
    
    Args:
        loss_curve: List of log loss values
        iterations: Total iterations
        title: Plot title
        filename: Optional filename to save
    
    Returns:
        fig: Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(np.linspace(0, iterations, len(loss_curve)), loss_curve,
           color=COLORS["trajectory"], linewidth=1.5)
    
    ax.set_xlabel("Iteration", fontsize=FONTS["label_size"])
    ax.set_ylabel("Log Loss", fontsize=FONTS["label_size"])
    ax.set_title(title, fontsize=FONTS["title_size"], fontweight='bold')
    ax.grid(False)
    
    if filename:
        plt.savefig(filename, dpi=PLOT_CONFIG["dpi"], bbox_inches='tight')
    
    return fig


def plot_steps_comparison(fm_model=None, samples_initial=None, samples_target=None,
                         title="Effect of Number of Steps", filename=None):
    """
    Plot comparison with different numbers of steps.
    
    Args:
        fm_model: Flow matching model
        samples_initial: Initial distribution samples
        samples_target: Target distribution samples
        title: Plot title
        filename: Optional filename to save
    
    Returns:
        fig: Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=PLOT_CONFIG["figsize_wide"])
    fig.suptitle(title, fontsize=FONTS["title_size"], fontweight='bold')
    
    for idx, num_steps in enumerate([100, 10]):
        with torch.no_grad():
            z0_test = samples_initial[:500]
            trajectory = fm_model.sample_ode(z0=z0_test, num_steps=num_steps)
        
        configure_distribution_plot(axes[idx])
        axes[idx].scatter(samples_initial[:, 0].cpu().numpy(),
                         samples_initial[:, 1].cpu().numpy(),
                         alpha=PLOT_CONFIG["alpha_distribution"],
                         s=PLOT_CONFIG["dot_size"],
                         color=COLORS["initial"])
        axes[idx].scatter(samples_target[:, 0].cpu().numpy(),
                         samples_target[:, 1].cpu().numpy(),
                         alpha=PLOT_CONFIG["alpha_distribution"],
                         s=PLOT_CONFIG["dot_size"],
                         color=COLORS["target"])
        axes[idx].scatter(trajectory[-1][:, 0].cpu().numpy(),
                         trajectory[-1][:, 1].cpu().numpy(),
                         alpha=PLOT_CONFIG["alpha_generated"],
                         s=PLOT_CONFIG["dot_size"],
                         color=COLORS["generated"])
        axes[idx].set_title(f'N={num_steps} steps', fontsize=FONTS["label_size"])
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=PLOT_CONFIG["dpi"], bbox_inches='tight')
    
    return fig


def plot_rf_comparison(rect_flow_1, rect_flow_2, samples_initial, samples_target,
                       title="1-RF vs 2-RF Comparison", filename=None):
    """
    Plot comparison between 1-RF and 2-RF.
    
    Args:
        rect_flow_1: 1-Rectified Flow model
        rect_flow_2: 2-Rectified Flow model
        samples_initial: Initial distribution samples
        samples_target: Target distribution samples
        title: Plot title
        filename: Optional filename to save
    
    Returns:
        fig: Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(title, fontsize=FONTS["title_size"], fontweight='bold')
    
    for idx, num_steps in enumerate([100, 1]):
        # 1-RF
        with torch.no_grad():
            z0_test = samples_initial[:500]
            traj_1rf = rect_flow_1.sample_ode(z0=z0_test, num_steps=num_steps)
        
        configure_distribution_plot(axes[0, idx])
        axes[0, idx].scatter(samples_initial[:, 0].cpu().numpy(),
                            samples_initial[:, 1].cpu().numpy(),
                            alpha=PLOT_CONFIG["alpha_distribution"],
                            s=PLOT_CONFIG["dot_size"],
                            color=COLORS["initial"])
        axes[0, idx].scatter(samples_target[:, 0].cpu().numpy(),
                            samples_target[:, 1].cpu().numpy(),
                            alpha=PLOT_CONFIG["alpha_distribution"],
                            s=PLOT_CONFIG["dot_size"],
                            color=COLORS["target"])
        axes[0, idx].scatter(traj_1rf[-1][:, 0].cpu().numpy(),
                            traj_1rf[-1][:, 1].cpu().numpy(),
                            alpha=PLOT_CONFIG["alpha_generated"],
                            s=PLOT_CONFIG["dot_size"],
                            color=COLORS["generated"])
        axes[0, idx].set_title(f'1-RF: N={num_steps} steps',
                              fontsize=FONTS["label_size"])
        
        # 2-RF
        with torch.no_grad():
            traj_2rf = rect_flow_2.sample_ode(z0=z0_test, num_steps=num_steps)
        
        configure_distribution_plot(axes[1, idx])
        axes[1, idx].scatter(samples_initial[:, 0].cpu().numpy(),
                            samples_initial[:, 1].cpu().numpy(),
                            alpha=PLOT_CONFIG["alpha_distribution"],
                            s=PLOT_CONFIG["dot_size"],
                            color=COLORS["initial"])
        axes[1, idx].scatter(samples_target[:, 0].cpu().numpy(),
                            samples_target[:, 1].cpu().numpy(),
                            alpha=PLOT_CONFIG["alpha_distribution"],
                            s=PLOT_CONFIG["dot_size"],
                            color=COLORS["target"])
        axes[1, idx].scatter(traj_2rf[-1][:, 0].cpu().numpy(),
                            traj_2rf[-1][:, 1].cpu().numpy(),
                            alpha=PLOT_CONFIG["alpha_generated"],
                            s=PLOT_CONFIG["dot_size"],
                            color=COLORS["generated"])
        axes[1, idx].set_title(f'2-RF: N={num_steps} steps',
                              fontsize=FONTS["label_size"])
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=PLOT_CONFIG["dpi"], bbox_inches='tight')
    
    return fig


# Import torch here to avoid circular imports
import torch
