# -*- coding: utf-8 -*-
"""
Flow Matching Training Script
Implements vanilla flow matching from unimodal gaussian to bimodal gaussian.
"""

import torch
import matplotlib.pyplot as plt
from data import create_distributions, create_data_pairs
from model import MLP, FlowMatching
from styles import (setup_plot_style, plot_loss_curve, 
                   plot_results_comparison, plot_steps_comparison,
                   plot_initial_distributions)
from config import TRAINING_CONFIG, MODEL_CONFIG, RANDOM_SEED, get_output_path, get_model_path
from utils import train_model, save_model, get_device, set_seed


def main():
    # Setup
    set_seed(RANDOM_SEED)
    setup_plot_style()
    device = get_device()
    print(f"Using device: {device}")
    
    # Get scenario
    scenario = TRAINING_CONFIG.get("scenario", "unimodal_to_bimodal")
    if scenario == "unimodal_to_bimodal":
        scenario_title = r"Unimodal $\rightarrow$ Bimodal"
    elif scenario == "bimodal_to_bimodal":
        scenario_title = r"Bimodal $\rightarrow$ Bimodal"
    else:
        scenario_title = scenario
    
    # Create distributions
    print(f"\nCreating distributions ({scenario_title})...")
    samples_0, samples_1 = create_distributions(device=device)
    print(f"Samples shape - pi_0: {samples_0.shape}, pi_1: {samples_1.shape}")
    
    # Plot initial distributions (just the starting scenario)
    print("Generating initial distributions plot...")
    dist_file = get_output_path("initial_distributions")
    plot_initial_distributions(samples_0, samples_1,
                              title=r"Initial Distributions: $\pi_0$ and $\pi_1$",
                              filename=dist_file)
    print(f"Saved: {dist_file}")
    
    # Create data pairs
    print("Creating data pairs...")
    x_pairs = create_data_pairs(samples_0, samples_1)
    print(f"Data pairs shape: {x_pairs.shape}")
    
    # Initialize model and optimizer
    print("\nInitializing model...")
    model = MLP(input_dim=MODEL_CONFIG["input_dim"], 
               hidden_num=MODEL_CONFIG["hidden_dim"])
    model.to(device)
    flow_matching = FlowMatching(model=model, num_steps=TRAINING_CONFIG["num_steps"])
    optimizer = torch.optim.Adam(model.parameters(), 
                                lr=TRAINING_CONFIG["learning_rate"])
    
    # Train
    print(f"\nTraining Flow Matching ({TRAINING_CONFIG['iterations_fm']} iterations)...")
    flow_matching, loss_curve = train_model(
        flow_matching, optimizer, x_pairs, 
        TRAINING_CONFIG["batch_size"], TRAINING_CONFIG["iterations_fm"],
        verbose=True, verbose_interval=2000
    )
    
    # Plot training loss
    print("\nGenerating training loss plot...")
    loss_file = get_output_path("fm_loss")
    plot_loss_curve(loss_curve, TRAINING_CONFIG["iterations_fm"],
                   title="Flow Matching Training Loss",
                   filename=loss_file)
    print(f"Saved: {loss_file}")
    
    # Visualize results with 100 steps
    print("Generating results visualization...")
    with torch.no_grad():
        z0_plot = samples_0[:1000]
        trajectory = flow_matching.sample_ode(z0=z0_plot, num_steps=100)
    
    results_file = get_output_path("fm_results")
    fig = plot_results_comparison(samples_0, samples_1, trajectory,
                                  title=f"Flow Matching: {scenario_title}")
    plt.savefig(results_file, dpi=200, bbox_inches='tight')
    print(f"Saved: {results_file}")
    
    # Test with different step counts
    print("Generating steps comparison...")
    steps_file = get_output_path("fm_steps")
    plot_steps_comparison(flow_matching, samples_0, samples_1,
                         title=f"Flow Matching: Effect of Number of Steps ({scenario_title})",
                         filename=steps_file)
    print(f"Saved: {steps_file}")
    
    # Save model
    model_file = get_model_path("fm_model")
    save_model(model, model_file)
    
    print("\nFlow Matching training completed!")
    plt.close('all')


if __name__ == "__main__":
    main()
