# -*- coding: utf-8 -*-
"""
Configuration and style settings for the Rectified Flow project.
"""

import os
from pathlib import Path

# ============================================================================
# PATH CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).parent
OUTPUT_ROOT = PROJECT_ROOT / "output"

# Create output subdirectories
OUTPUT_DIRS = {
    "flow_matching": OUTPUT_ROOT / "flow_matching",
    "flow_matching_plots": OUTPUT_ROOT / "flow_matching" / "plots",
    "flow_matching_models": OUTPUT_ROOT / "flow_matching" / "models",
    "rectified_flow": OUTPUT_ROOT / "rectified_flow",
    "1rf_plots": OUTPUT_ROOT / "rectified_flow" / "1rf_plots",
    "2rf_plots": OUTPUT_ROOT / "rectified_flow" / "2rf_plots",
    "rf_models": OUTPUT_ROOT / "rectified_flow" / "models",
    "animations": OUTPUT_ROOT / "animations",
    "comparison": OUTPUT_ROOT / "comparison",
}

# Create all directories if they don't exist
for directory in OUTPUT_DIRS.values():
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# COLOR CONFIGURATION
# ============================================================================

COLORS = {
    "initial": "#E32986",        # Purple for π₀
    "target": "#1C55EF",         # Blue for π₁
    "generated": "#251B67",      # Lime green for generated
    "trajectory": "#707070",     # Gray for trajectories
    "particles": "#251B67",      # Orange red for particles
}

# ============================================================================
# PLOT CONFIGURATION
# ============================================================================

PLOT_CONFIG = {
    "dpi": 200,
    "figsize": (8, 8),
    "figsize_wide": (14, 4),
    "figsize_comparison": (16, 8),
    "alpha_distribution": 0.15,
    "alpha_generated": 0.2,
    "alpha_trajectory": 0.3,
    "alpha_trajectory_anim": 0.2,
    "dot_size": 10,
    "trajectory_linewidth": 0.5,
    "trajectory_linewidth_anim": 0.5,
    "xlim": (-10, 10),
    "ylim": (-10, 10),
}

# ============================================================================
# FONT CONFIGURATION
# ============================================================================

FONTS = {
    "serif": "CMU Serif",
    "sans_serif": "CMU Sans Serif",
    "title_size": 14,
    "label_size": 12,
    "tick_size": 10,
    "legend_size": 10,
}

# ============================================================================
# DISTRIBUTION PARAMETERS
# ============================================================================

RANDOM_SEED = 100

# Choose scenario: "unimodal_to_bimodal" or "bimodal_to_bimodal"
SCENARIO = "bimodal_to_bimodal"

DISTRIBUTION_CONFIG = {
    "scenario": SCENARIO,
    "d": 5.0,
    "m": 12.0,
    "initial_var": 0.5,
    "target_var": 0.5,
    "n_samples": 10000,
    "n_samples_per_mode": 5000,
    "bimodal_separation": 8.0,  # For bimodal-to-bimodal scenario (modes in separate quadrants)
}

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

MODEL_CONFIG = {
    "input_dim": 2,
    "hidden_dim": 128,
}

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

TRAINING_CONFIG = {
    "batch_size": 256,
    "learning_rate": 5e-3,
    "iterations_fm": 10000,
    "iterations_1rf": 20000,
    "iterations_2rf": 20000,
    "num_steps": 100,
}

# ============================================================================
# ANIMATION CONFIGURATION
# ============================================================================

ANIMATION_CONFIG = {
    "num_particles": 100,
    "num_steps": 100,
    "fps": 30,
    "dpi": 100,
    "bitrate": 1800,
}

# ============================================================================
# FILE NAMES
# ============================================================================

FILE_NAMES = {
    # Flow Matching
    "fm_loss": "flow_matching_loss.png",
    "fm_results": "flow_matching_results.png",
    "fm_steps": "flow_matching_steps_comparison.png",
    "fm_model": "flow_matching_model.pt",
    "fm_animation": "flow_matching_animation.mp4",
    
    # Rectified Flow 1
    "1rf_loss": "1rf_loss.png",
    "1rf_results": "1rf_results.png",
    "1rf_model": "1rf_model.pt",
    "1rf_animation": "1rf_animation.mp4",
    
    # Rectified Flow 2
    "2rf_loss": "2rf_loss.png",
    "2rf_results": "2rf_results.png",
    "2rf_model": "2rf_model.pt",
    "2rf_animation": "2rf_animation.mp4",
    
    # Comparison
    "rf_comparison": "rf_comparison.png",
    "comparison_animation": "comparison_animation.mp4",
    
    # Initial Distributions
    "initial_distributions": "initial_distributions.png",
}

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def get_output_path(key: str, subfolder: str = None) -> str:
    """
    Get full output path for a file.
    
    Args:
        key: Key from FILE_NAMES dict
        subfolder: Optional subdirectory in output folder
    
    Returns:
        Full path to output file
    """
    filename = FILE_NAMES.get(key, key)
    
    if "fm_" in key:
        base_dir = OUTPUT_DIRS["flow_matching_plots"]
    elif "1rf_" in key:
        base_dir = OUTPUT_DIRS["1rf_plots"]
    elif "2rf_" in key:
        base_dir = OUTPUT_DIRS["2rf_plots"]
    elif "animation" in key:
        base_dir = OUTPUT_DIRS["animations"]
    elif "comparison" in key:
        base_dir = OUTPUT_DIRS["comparison"]
    else:
        base_dir = OUTPUT_ROOT
    
    return str(base_dir / filename)

def get_model_path(key: str) -> str:
    """Get path for model files."""
    filename = FILE_NAMES.get(key, key)
    
    if "fm_" in key:
        return str(OUTPUT_DIRS["flow_matching_models"] / filename)
    elif "1rf_" in key or "2rf_" in key:
        return str(OUTPUT_DIRS["rf_models"] / filename)
    else:
        return str(OUTPUT_ROOT / filename)
