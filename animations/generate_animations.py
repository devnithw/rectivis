# -*- coding: utf-8 -*-
"""
Animation Generation Script
Creates mp4 videos for Flow Matching and Rectified Flow showing particle trajectories.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from animations.data import create_distributions
from animations.model import MLP, ODEModel
from animations.config import get_model_path, get_output_path, OUTPUT_DIRS, ANIMATION_CONFIG, COLORS, PLOT_CONFIG, RANDOM_SEED
from animations.utils import get_device, load_model, set_seed
from animations.styles import setup_plot_style, configure_distribution_plot


def create_animation(trajectory, initial_samples, target_samples, 
                    title, output_file, fps=30):
    """
    Create animation showing particle trajectories.
    
    Args:
        trajectory: tensor of shape (num_frames, num_particles, 2)
        initial_samples: sample points from initial distribution
        target_samples: sample points from target distribution
        title: title for the animation
        output_file: output mp4 filename
        fps: frames per second
    """
    trajectory = trajectory.cpu().numpy()
    initial_samples = initial_samples.cpu().numpy()
    target_samples = target_samples.cpu().numpy()
    
    num_frames = trajectory.shape[0]
    num_particles = trajectory.shape[1]
    
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    
    # Set up the plot
    configure_distribution_plot(ax)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Plot initial and target distributions (background)
    ax.scatter(initial_samples[:, 0], initial_samples[:, 1], 
              alpha=PLOT_CONFIG["alpha_distribution"], 
              s=PLOT_CONFIG["dot_size"],
              color=COLORS["initial"])
    ax.scatter(target_samples[:, 0], target_samples[:, 1], 
              alpha=PLOT_CONFIG["alpha_distribution"], 
              s=PLOT_CONFIG["dot_size"],
              color=COLORS["target"])

    
    # Initialize plot elements
    scatter = ax.scatter([], [], s=50, color=COLORS["particles"], alpha=0.8)
    lines = [ax.plot([], [], alpha=PLOT_CONFIG["alpha_trajectory_anim"], 
                    linewidth=PLOT_CONFIG["trajectory_linewidth_anim"], 
                    color=COLORS["trajectory"])[0] 
             for _ in range(num_particles)]
    
    # Time text
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, 
                       fontsize=12, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def init():
        scatter.set_offsets(np.empty((0, 2)))
        for line in lines:
            line.set_data([], [])
        time_text.set_text('')
        return [scatter] + lines + [time_text]
    
    def animate(frame):
        # Current particle positions
        positions = trajectory[frame]
        scatter.set_offsets(positions)
        
        # Update trajectories
        for i in range(num_particles):
            particle_trajectory = trajectory[:frame+1, i, :]
            lines[i].set_data(particle_trajectory[:, 0], particle_trajectory[:, 1])
        
        # Update time
        t_val = frame / (num_frames - 1)
        time_text.set_text(f't = {t_val:.2f}')
        
        return [scatter] + lines + [time_text]
    
    print(f"Creating animation: {output_file}")
    print(f"  Frames: {num_frames}, Particles: {num_particles}, FPS: {fps}")
    
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                  frames=num_frames, interval=1000/fps,
                                  blit=True, repeat=False)
    
    # Save animation
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps, bitrate=ANIMATION_CONFIG["bitrate"])
    anim.save(output_file, writer=writer, dpi=ANIMATION_CONFIG["dpi"])
    print(f"Saved: {output_file}")
    
    plt.close(fig)


def generate_flow_matching_animation():
    """Generate animation for Flow Matching."""
    device = get_device()
    print("\n" + "="*60)
    print("Generating Flow Matching Animation")
    print("="*60)
    
    # Load model
    model_path = get_model_path("fm_model")
    
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found. Please run train_flow_matching.py first.")
        return
    
    model = load_model(MLP, model_path, device=device)
    
    # Create distributions
    samples_0, samples_1 = create_distributions(device=device)
    
    # Sample particles and get trajectory
    z0_particles = samples_0[:ANIMATION_CONFIG["num_particles"]]
    ode_model = ODEModel(model, num_steps=ANIMATION_CONFIG["num_steps"])
    trajectory = ode_model.get_trajectory(z0_particles, num_steps=ANIMATION_CONFIG["num_steps"])
    
    # Create animation
    output_file = get_output_path("fm_animation")
    create_animation(trajectory, samples_0, samples_1,
                    r'Flow Matching: Unimodal $\rightarrow$ Bimodal',
                    output_file, fps=ANIMATION_CONFIG["fps"])


def generate_rectified_flow_animations():
    """Generate animations for 1-RF and 2-RF."""
    device = get_device()
    print("\n" + "="*60)
    print("Generating Rectified Flow Animations")
    print("="*60)
    
    # Create distributions
    samples_0, samples_1 = create_distributions(device=device)
    
    # Load 1-RF model
    model_1_path = get_model_path("1rf_model")
    
    if not os.path.exists(model_1_path):
        print(f"Error: {model_1_path} not found. Please run train_rectified_flow.py first.")
        return
    
    model_1 = load_model(MLP, model_1_path, device=device)
    
    # Generate 1-RF animation
    z0_particles = samples_0[:ANIMATION_CONFIG["num_particles"]]
    ode_model_1 = ODEModel(model_1, num_steps=ANIMATION_CONFIG["num_steps"])
    trajectory_1 = ode_model_1.get_trajectory(z0_particles, num_steps=ANIMATION_CONFIG["num_steps"])
    
    output_file_1 = get_output_path("1rf_animation")
    create_animation(trajectory_1, samples_0, samples_1,
                    r'1-Rectified Flow: Unimodal $\rightarrow$ Bimodal',
                    output_file_1, fps=ANIMATION_CONFIG["fps"])
    
    # Load 2-RF model
    model_2_path = get_model_path("2rf_model")
    
    if not os.path.exists(model_2_path):
        print(f"Error: {model_2_path} not found. Please run train_rectified_flow.py first.")
        return
    
    model_2 = load_model(MLP, model_2_path, device=device)
    
    # Generate 2-RF animation
    ode_model_2 = ODEModel(model_2, num_steps=ANIMATION_CONFIG["num_steps"])
    trajectory_2 = ode_model_2.get_trajectory(z0_particles, num_steps=ANIMATION_CONFIG["num_steps"])
    
    output_file_2 = get_output_path("2rf_animation")
    create_animation(trajectory_2, samples_0, samples_1,
                    r'2-Rectified Flow (Straightened): Bimodal $\rightarrow$ Bimodal',
                    output_file_2, fps=ANIMATION_CONFIG["fps"])


def generate_combined_comparison_animation():
    """Generate side-by-side comparison animation."""
    device = get_device()
    print("\n" + "="*60)
    print("Generating Combined Comparison Animation")
    print("="*60)
    
    # Create distributions
    samples_0, samples_1 = create_distributions(device=device)
    
    # Load Flow Matching model
    fm_path = get_model_path("fm_model")
    if not os.path.exists(fm_path):
        print(f"Error: {fm_path} not found.")
        return
    fm_model = load_model(MLP, fm_path, device=device)
    
    # Load 2-RF model
    rf_path = get_model_path("2rf_model")
    if not os.path.exists(rf_path):
        print(f"Error: {rf_path} not found.")
        return
    rf_model = load_model(MLP, rf_path, device=device)
    
    # Get trajectories
    z0_particles = samples_0[:ANIMATION_CONFIG["num_particles"]]
    
    fm_ode = ODEModel(fm_model, num_steps=ANIMATION_CONFIG["num_steps"])
    fm_trajectory = fm_ode.get_trajectory(z0_particles, num_steps=ANIMATION_CONFIG["num_steps"])
    
    rf_ode = ODEModel(rf_model, num_steps=ANIMATION_CONFIG["num_steps"])
    rf_trajectory = rf_ode.get_trajectory(z0_particles, num_steps=ANIMATION_CONFIG["num_steps"])
    
    # Create side-by-side animation
    fm_trajectory = fm_trajectory.cpu().numpy()
    rf_trajectory = rf_trajectory.cpu().numpy()
    initial_samples = samples_0.cpu().numpy()
    target_samples = samples_1.cpu().numpy()
    
    num_frames = fm_trajectory.shape[0]
    num_particles = fm_trajectory.shape[1]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), dpi=100)
    
    for ax, trajectory, title in [(ax1, fm_trajectory, 'Flow Matching'),
                                   (ax2, rf_trajectory, '2-Rectified Flow')]:
        configure_distribution_plot(ax)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Plot distributions
        ax.scatter(initial_samples[:, 0], initial_samples[:, 1],
                  alpha=PLOT_CONFIG["alpha_distribution"], 
                  s=PLOT_CONFIG["dot_size"],
                  color=COLORS["initial"])
        ax.scatter(target_samples[:, 0], target_samples[:, 1],
                  alpha=PLOT_CONFIG["alpha_distribution"],
                  s=PLOT_CONFIG["dot_size"],
                  color=COLORS["target"])

    
    # Animation state
    scatters = [ax1.scatter([], [], s=50, color=COLORS["particles"], alpha=0.8),
               ax2.scatter([], [], s=50, color=COLORS["particles"], alpha=0.8)]
    lines_1 = [ax1.plot([], [], alpha=PLOT_CONFIG["alpha_trajectory_anim"], 
                       linewidth=PLOT_CONFIG["trajectory_linewidth_anim"], 
                       color=COLORS["trajectory"])[0]
              for _ in range(num_particles)]
    lines_2 = [ax2.plot([], [], alpha=PLOT_CONFIG["alpha_trajectory_anim"],
                       linewidth=PLOT_CONFIG["trajectory_linewidth_anim"],
                       color=COLORS["trajectory"])[0]
              for _ in range(num_particles)]
    
    time_text = fig.text(0.5, 0.02, '', ha='center', fontsize=12,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def init():
        for scatter in scatters:
            scatter.set_offsets(np.empty((0, 2)))
        for line in lines_1 + lines_2:
            line.set_data([], [])
        time_text.set_text('')
        return scatters + lines_1 + lines_2 + [time_text]
    
    def animate(frame):
        # FM animation
        scatters[0].set_offsets(fm_trajectory[frame])
        for i in range(num_particles):
            lines_1[i].set_data(fm_trajectory[:frame+1, i, 0], 
                               fm_trajectory[:frame+1, i, 1])
        
        # RF animation
        scatters[1].set_offsets(rf_trajectory[frame])
        for i in range(num_particles):
            lines_2[i].set_data(rf_trajectory[:frame+1, i, 0],
                               rf_trajectory[:frame+1, i, 1])
        
        t_val = frame / (num_frames - 1)
        time_text.set_text(f't = {t_val:.2f}')
        
        return scatters + lines_1 + lines_2 + [time_text]
    
    output_file = get_output_path("comparison_animation")
    print(f"Creating comparison animation: {output_file}")
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                  frames=num_frames, interval=1000/ANIMATION_CONFIG["fps"],
                                  blit=True, repeat=False)
    
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=ANIMATION_CONFIG["fps"], bitrate=ANIMATION_CONFIG["bitrate"])
    anim.save(output_file, writer=writer, dpi=ANIMATION_CONFIG["dpi"])
    print(f"Saved: {output_file}")
    
    plt.close(fig)


if __name__ == "__main__":
    set_seed(RANDOM_SEED)
    setup_plot_style()
    print("Animation Generation Script")
    print("="*60)
    
    # Generate all animations
    generate_flow_matching_animation()
    generate_rectified_flow_animations()
    generate_combined_comparison_animation()
    
    print("\n" + "="*60)
    print("All animations completed!")
    print("="*60)
    print("\nGenerated files:")
    print(f"  - {get_output_path('fm_animation')}")
    print(f"  - {get_output_path('1rf_animation')}")
    print(f"  - {get_output_path('2rf_animation')}")
    print(f"  - {get_output_path('comparison_animation')}")
