# -*- coding: utf-8 -*-
"""
Rectified Flow Training Script
Implements rectified flow with reflow procedure from unimodal gaussian to bimodal gaussian.
"""

import torch
import matplotlib.pyplot as plt
import copy
from data import create_distributions, create_data_pairs
from model import MLP, RectifiedFlow
from styles import (setup_plot_style, plot_loss_curve, 
                   plot_results_comparison, plot_rf_comparison)
from config import TRAINING_CONFIG, MODEL_CONFIG, RANDOM_SEED, get_output_path, get_model_path
from utils import train_model, save_model, get_device, copy_model, set_seed


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
    
    # ============ Stage 1: 1-Rectified Flow ============
    print("\n" + "="*60)
    print("Stage 1: Training 1-Rectified Flow")
    print("="*60)
    
    # Create random independent pairings
    x_pairs = create_data_pairs(samples_0, samples_1)
    print(f"Data pairs shape: {x_pairs.shape}")
    
    # Initialize model and optimizer
    model_1 = MLP(input_dim=MODEL_CONFIG["input_dim"], 
                 hidden_num=MODEL_CONFIG["hidden_dim"])
    model_1.to(device)
    rect_flow_1 = RectifiedFlow(model=model_1, num_steps=TRAINING_CONFIG["num_steps"])
    optimizer_1 = torch.optim.Adam(model_1.parameters(), 
                                  lr=TRAINING_CONFIG["learning_rate"])
    
    # Train 1-RF
    print(f"Training 1-Rectified Flow ({TRAINING_CONFIG['iterations_1rf']} iterations)...")
    rect_flow_1, loss_curve_1 = train_model(
        rect_flow_1, optimizer_1, x_pairs, 
        TRAINING_CONFIG["batch_size"], TRAINING_CONFIG["iterations_1rf"],
        verbose=True, verbose_interval=2000
    )
    
    # Plot training loss for 1-RF
    print("\nGenerating 1-RF loss plot...")
    loss_1_file = get_output_path("1rf_loss")
    plot_loss_curve(loss_curve_1, TRAINING_CONFIG["iterations_1rf"],
                   title="1-Rectified Flow Training Loss",
                   filename=loss_1_file)
    print(f"Saved: {loss_1_file}")
    
    # Visualize 1-RF results
    print("Generating 1-RF results visualization...")
    with torch.no_grad():
        z0_plot = samples_0[:1000]
        trajectory_1 = rect_flow_1.sample_ode(z0=z0_plot, num_steps=100)
    
    results_1_file = get_output_path("1rf_results")
    fig = plot_results_comparison(samples_0, samples_1, trajectory_1,
                                  title="1-Rectified Flow")
    plt.savefig(results_1_file, dpi=200, bbox_inches='tight')
    print(f"Saved: {results_1_file}")
    
    # Save 1-RF model
    model_1_file = get_model_path("1rf_model")
    save_model(model_1, model_1_file)
    
    # ============ Stage 2: Reflow to 2-Rectified Flow ============
    print("\n" + "="*60)
    print("Stage 2: Reflow - Training 2-Rectified Flow")
    print("="*60)
    
    # Generate coupling from 1-RF
    print("Generating reflow coupling from 1-RF...")
    z0_rf = samples_0.detach().clone()
    with torch.no_grad():
        traj_1 = rect_flow_1.sample_ode(z0=z0_rf, num_steps=100)
    z1_rf = traj_1[-1].detach().clone()
    z_pairs_reflow_1 = torch.stack([z0_rf, z1_rf], dim=1)
    print(f"Reflow coupling shape: {z_pairs_reflow_1.shape}")
    
    # Initialize 2-RF model from 1-RF
    model_2 = MLP(input_dim=MODEL_CONFIG["input_dim"], 
                 hidden_num=MODEL_CONFIG["hidden_dim"])
    model_2.load_state_dict(copy_model(model_1.state_dict()))
    model_2.to(device)
    rect_flow_2 = RectifiedFlow(model=model_2, num_steps=TRAINING_CONFIG["num_steps"])
    optimizer_2 = torch.optim.Adam(model_2.parameters(), 
                                  lr=TRAINING_CONFIG["learning_rate"])
    
    # Train 2-RF
    print(f"Training 2-Rectified Flow ({TRAINING_CONFIG['iterations_2rf']} iterations)...")
    rect_flow_2, loss_curve_2 = train_model(
        rect_flow_2, optimizer_2, z_pairs_reflow_1, 
        TRAINING_CONFIG["batch_size"], TRAINING_CONFIG["iterations_2rf"],
        verbose=True, verbose_interval=2000
    )
    
    # Plot training loss for 2-RF
    print("\nGenerating 2-RF loss plot...")
    loss_2_file = get_output_path("2rf_loss")
    plot_loss_curve(loss_curve_2, TRAINING_CONFIG["iterations_2rf"],
                   title="2-Rectified Flow Training Loss",
                   filename=loss_2_file)
    print(f"Saved: {loss_2_file}")
    
    # Visualize 2-RF results
    print("Generating 2-RF results visualization...")
    with torch.no_grad():
        trajectory_2 = rect_flow_2.sample_ode(z0=z0_plot, num_steps=100)
    
    results_2_file = get_output_path("2rf_results")
    fig = plot_results_comparison(samples_0, samples_1, trajectory_2,
                                  title="2-Rectified Flow (After Reflow)")
    plt.savefig(results_2_file, dpi=200, bbox_inches='tight')
    print(f"Saved: {results_2_file}")
    
    # Save 2-RF model
    model_2_file = get_model_path("2rf_model")
    save_model(model_2, model_2_file)
    
    # ============ Comparison ============
    print("\n" + "="*60)
    print("Stage 3: Comparing 1-RF vs 2-RF with Different Step Counts")
    print("="*60)
    
    comparison_file = get_output_path("rf_comparison")
    
    plot_rf_comparison(rect_flow_1, rect_flow_2, samples_0, samples_1,
                      title="Rectified Flow: 1-RF vs 2-RF",
                      filename=comparison_file)
    print(f"Saved: {comparison_file}")
    
    print("\nRectified Flow training completed!")
    plt.close('all')


if __name__ == "__main__":
    main()
