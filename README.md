# Rectified Flow Implementation with Flow Matching Comparison

This project implements both vanilla **Flow Matching** and **Rectified Flow** with a reflow procedure, showing the transformation from a unimodal Gaussian to a bimodal Gaussian distribution. The code is fully modularized with centralized configuration, styling, and data handling.

## Overview

### Distributions
- **Initial (π₀)**: Unimodal Gaussian centered at (-8, 0)
- **Target (π₁)**: Bimodal Gaussian with two modes at (8, 4) and (8, -4)
- Distributions are spatially separated to clearly visualize flow trajectories

### Key Concepts

#### 1. **Flow Matching (Vanilla)**
- Uses simple linear interpolation: z_t = (1-t)·z₀ + t·z₁
- Target velocity: v = z₁ - z₀
- Trains a neural network to predict velocity field
- **Scripts**: `train_flow_matching.py`

#### 2. **Rectified Flow** 
- More sophisticated approach with two stages
- **Stage 1 (1-RF)**: Train on independent couplings of (z₀, z₁)
- **Stage 2 (2-RF)**: Reflow procedure creates deterministic couplings by sampling through 1-RF
- Results in **straightened trajectories** → efficient one-step generation
- **Scripts**: `train_rectified_flow.py`

#### 3. **Animations**
- Shows particle trajectories with semi-transparent flow lines
- Compares Flow Matching vs Rectified Flow performance
- **Scripts**: `generate_animations.py`

## Project Structure

### Core Architecture (Modularized)

```
rect-flow/
├── config.py                      # Central configuration and paths
├── data.py                        # Distribution creation utilities
├── model.py                       # Neural network and ODE models
├── styles.py                      # Plotting utilities and styling
├── utils.py                       # Common training/model utilities
│
├── train_flow_matching.py         # Flow Matching training script
├── train_rectified_flow.py        # Rectified Flow training script
├── generate_animations.py         # Animation generation script
│
├── README.md                      # This file
└── output/                        # Output directory (created automatically)
    ├── flow_matching/
    │   ├── plots/
    │   └── models/
    ├── rectified_flow/
    │   ├── 1rf_plots/
    │   ├── 2rf_plots/
    │   └── models/
    ├── animations/
    └── comparison/
```

### Output Structure

All outputs are automatically organized in the `output/` folder:

- **`output/flow_matching/plots/`**
  - `flow_matching_loss.png`
  - `flow_matching_results.png`
  - `flow_matching_steps_comparison.png`
  
- **`output/flow_matching/models/`**
  - `flow_matching_model.pt`
  
- **`output/rectified_flow/1rf_plots/`**
  - `1rf_loss.png`
  - `1rf_results.png`
  
- **`output/rectified_flow/2rf_plots/`**
  - `2rf_loss.png`
  - `2rf_results.png`
  - `rf_comparison.png`
  
- **`output/rectified_flow/models/`**
  - `1rf_model.pt`
  - `2rf_model.pt`
  
- **`output/animations/`**
  - `flow_matching_animation.mp4`
  - `1rf_animation.mp4`
  - `2rf_animation.mp4`
  - `comparison_animation.mp4`

## Modular Components

### 1. **config.py** - Central Configuration Hub
Centralized configuration with easy customization:
- **Colors**: Purple (initial), Blue (target), Green (generated), Gray (trajectories)
- **Plot Settings**: DPI=200, fonts (CMU Serif/Sans-Serif), transparency levels
- **Hyperparameters**: Batch size, learning rate, iterations
- **Paths**: Automatic output directory management

### 2. **data.py** - Data Generation
- `create_distributions()`: Generate initial and target distributions
- `create_data_pairs()`: Create random sample pairings

### 3. **model.py** - Neural Networks & ODE Solvers
- `MLP`: 3-layer neural network for velocity prediction
- `FlowMatching`: Flow matching model wrapper
- `RectifiedFlow`: Rectified flow model wrapper
- `ODEModel`: Generic ODE solver for inference

### 4. **styles.py** - Unified Plotting
Centralized styling for all plots:
- `configure_distribution_plot()`: Set standard axes/limits
- `plot_loss_curve()`: Training loss visualization
- `plot_results_comparison()`: 3-panel results comparison
- `plot_steps_comparison()`: Effect of number of steps
- `plot_rf_comparison()`: 1-RF vs 2-RF comparison
- CMU Sans Serif titles, CMU Serif labels
- No tick marks/labels for cleaner plots

### 5. **utils.py** - Training Utilities
- `train_model()`: Generic training loop
- `save_model()` / `load_model()`: Model persistence
- `get_device()`: Automatic GPU/CPU selection
- `print_config()`: Pretty-print configuration

## Setup & Dependencies

### Requirements
- Python 3.8+
- PyTorch (with CUDA support recommended)
- NumPy
- Matplotlib
- FFmpeg (for animation generation)

### Installation

```bash
# Install FFmpeg (if not already installed)
# On macOS:
brew install ffmpeg

# On Ubuntu/Debian:
sudo apt-get install ffmpeg

# Python packages (in your pytorch conda environment)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install matplotlib numpy
```

## Usage

### Step 1: Train Flow Matching Model

```bash
python train_flow_matching.py
```

**Output files** (automatically organized in `output/`):
- Training loss plot
- Qualitative results (3-panel comparison)
- Steps comparison (100 vs 10 steps)
- Trained model weights

### Step 2: Train Rectified Flow Model

```bash
python train_rectified_flow.py
```

**Output files**:
- 1-RF training loss and results
- 2-RF training loss and results
- 1-RF vs 2-RF comparison plot
- Trained model weights (1-RF and 2-RF)

### Step 3: Generate Animations

```bash
python generate_animations.py
```

**Prerequisites**: Run Steps 1 & 2 first.

**Output videos**:
- `flow_matching_animation.mp4`
- `1rf_animation.mp4`
- `2rf_animation.mp4` (straightened trajectories)
- `comparison_animation.mp4` (side-by-side FM vs 2-RF)

## Customization

### Change Color Scheme

Edit `config.py`:
```python
COLORS = {
    "initial": "#8B008B",        # Initial distribution (purple)
    "target": "#0000CD",         # Target distribution (blue)
    "generated": "#32CD32",      # Generated samples (green)
    "trajectory": "#808080",     # Flow trajectories (gray)
    "particles": "#FF4500",      # Animation particles
}
```

### Adjust Plotting

Edit `config.py` PLOT_CONFIG:
```python
PLOT_CONFIG = {
    "dpi": 200,                  # Resolution
    "alpha_distribution": 0.15,  # Transparency for distributions
    "dot_size": 10,              # Point size
    # ... more options
}
```

### Modify Training Parameters

Edit `config.py` TRAINING_CONFIG:
```python
TRAINING_CONFIG = {
    "batch_size": 256,
    "learning_rate": 5e-3,
    "iterations_fm": 10000,
    "iterations_1rf": 10000,
    "iterations_2rf": 20000,
}
```

### Change Distributions

Edit `config.py` DISTRIBUTION_CONFIG:
```python
DISTRIBUTION_CONFIG = {
    "d": 8.0,                    # Distance of bimodal means
    "n_samples": 10000,          # Number of samples
    "initial_var": 0.5,          # Initial distribution variance
    "target_var": 0.5,           # Target distribution variance
}
```

## Model Architecture

All models use the same simple 3-layer MLP:

```python
Input: [x, y, t] (position + time)
  ↓
Hidden Layer: 128 units, tanh activation
  ↓
Hidden Layer: 128 units, tanh activation
  ↓
Output: [vx, vy] (velocity prediction)
```

## Visualizations

### Plots (DPI=200)
- **CMU Sans Serif** for titles (bold)
- **CMU Serif** for axis labels
- No grid lines
- No tick marks/numbers
- Purple initial, Blue target, Green generated distributions
- Gray semi-transparent trajectory lines

### Animations
- 100 particles showing flow lines
- Semi-transparent trajectories (α=0.2)
- Purple/Blue background distributions
- Time indicator (t ∈ [0, 1])
- 30 FPS, MP4 format

## Performance Tips

### Memory Issues
- Reduce `batch_size` in `config.py`
- Reduce `num_particles` in `ANIMATION_CONFIG`
- Use GPU for faster training

### Speed Optimization
- Verify CUDA: `torch.cuda.is_available()`
- Increase `learning_rate` (carefully)
- Use fewer sampling steps during training

### Better Results
- Increase training iterations in `config.py`
- Ensure distributions are well-separated
- Adjust `learning_rate` and `batch_size`

## Key Insights

### Flow Matching
✓ Simple and interpretable  
✓ Straightforward training  
✗ Curved trajectories → many steps needed  

### Rectified Flow
✓ Straightened trajectories after reflow  
✓ Efficient: works with very few steps (even 1!)  
✓ Theoretical trajectory straightness guarantees  
✗ Two-stage training (more compute)  

### The Reflow Miracle
The key innovation: by training on deterministic couplings (generated from the first model), we get straight trajectories → one-step generation! 🚀

## References

- **Rectified Flow**: [Flow Straight and Fast](https://arxiv.org/abs/2209.03003)
- **Flow Matching**: [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)

## Code Organization Benefits

✓ **Centralized Configuration**: Change colors/settings in one place  
✓ **Unified Styling**: Consistent plots across all scripts  
✓ **Reusable Components**: Easy to extend or modify  
✓ **Clean Separation**: Data, models, plots, training are independent  
✓ **Automatic Output Management**: Organized folder structure  
✓ **Higher Resolution**: All outputs at DPI=200  

## License

This is an educational implementation based on the tutorial code structure.

