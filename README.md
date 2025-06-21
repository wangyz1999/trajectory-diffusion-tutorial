# 2D Trajectory Diffusion Model

A complete machine learning project for generating 2D trajectories using diffusion models with UNet1D architecture. This project treats 2D trajectories as sequences over time and uses mathematical patterns like sine waves, spirals, circles, lemniscates, and cardioids.

## Project Structure

```
trajectory-diffusion-tutorial/
├── generate_dataset.py     # Generate mathematical patterns and save dataset
├── dataset.py             # PyTorch dataset classes
├── model.py               # UNet1D neural network architecture
├── trainer.py             # Training script with experiment management
├── test.py                # Evaluation script with metrics and plots
├── inference.py           # Generate new trajectories using trained model
├── utils.py               # Utility functions for experiment management
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── data/                  # Dataset directory (created automatically)
└── output/                # Experiment outputs directory (created automatically)
    └── trajectory_diffusion_XXX/  # Individual experiment directories
```

## Features

- **Mathematical Pattern Generation**: Generates sine waves, spirals, circles, lemniscates, and cardioids
- **Diffusion Model**: Uses DDPM (Denoising Diffusion Probabilistic Model) with UNet1D architecture
- **Experiment Management**: Automatic experiment indexing and organization
- **Comprehensive Evaluation**: Multiple metrics and visualizations
- **Trajectory Generation**: Generate new trajectories and interpolations
- **Structured Codebase**: Modular design with clear separation of concerns

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd trajectory-diffusion-tutorial
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Generate Dataset

First, generate the mathematical trajectory patterns:

```bash
python generate_dataset.py
```

This creates a `data/` directory with:
- `trajectories.npy`: Trajectory data [N, sequence_length, 2]
- `labels.json`: Pattern type labels
- `metadata.json`: Dataset metadata
- `sample_trajectories.png`: Visualization of sample patterns

### 2. Train the Model

Train the diffusion model:

```bash
python trainer.py
```

This will:
- Create an experiment directory: `output/trajectory_diffusion_000/`
- Train the UNet1D model with DDPM scheduler
- Save model checkpoints and training curves
- Track training/validation losses

### 3. Evaluate the Model

Evaluate the trained model on test data:

```bash
python test.py --exp_idx 0
```

This generates:
- Reconstruction metrics
- Sample quality metrics
- Comparison plots between real and generated trajectories
- Distribution analysis plots
- Trajectory animations

### 4. Generate New Trajectories

Generate new trajectories using the trained model:

```bash
python inference.py --exp_idx 0 --num_samples 20 --create_plots --create_animations
```

Options:
- `--exp_idx`: Experiment index to use
- `--num_samples`: Number of trajectories to generate
- `--seed`: Random seed for reproducible generation
- `--create_plots`: Create visualization plots
- `--create_animations`: Create trajectory animations
- `--interpolate`: Generate trajectory interpolations

## Model Architecture

### UNet1D
- **Input**: Noisy trajectory [batch, 2, sequence_length]
- **Output**: Predicted noise [batch, 2, sequence_length]
- **Features**:
  - Sinusoidal time embeddings
  - Residual blocks with time conditioning
  - Self-attention at multiple scales
  - U-Net architecture with skip connections

### Diffusion Process
- **Forward Process**: Gradually add Gaussian noise to trajectories
- **Reverse Process**: Learn to denoise trajectories step by step
- **Scheduler**: DDPM with linear beta schedule
- **Training**: Predict noise at random timesteps

## Dataset

### Pattern Types
1. **Sine Wave**: `y = A * sin(f * t + φ)`
2. **Spiral**: Archimedean spiral `r = a + b * t`
3. **Circle**: `x = r * cos(t), y = r * sin(t)`
4. **Lemniscate**: Figure-8 pattern
5. **Cardioid**: Heart-shaped curve

### Data Properties
- **Sequence Length**: 100 points per trajectory
- **Normalization**: All trajectories normalized to [-1, 1]
- **Noise**: 30% of trajectories have added Gaussian noise
- **Samples**: 1000 samples per pattern type (5000 total)

## Experiment Management

The project uses automatic experiment indexing:

```
output/
├── trajectory_diffusion_000/  # First experiment
│   ├── config.json           # Experiment configuration
│   ├── best_model.pth         # Best model checkpoint
│   ├── training_curves.png    # Training progress plots
│   └── evaluation_results.json # Evaluation metrics
├── trajectory_diffusion_001/  # Second experiment
└── ...
```

## Evaluation Metrics

### Reconstruction Metrics
- Mean Squared Error (MSE)
- Reconstruction loss at different noise levels

### Sample Quality Metrics
- Mean Absolute Error (MAE)
- Fréchet Distance
- Endpoint Distance
- Trajectory Length Comparison
- Velocity Distribution Analysis

## Customization

### Modify Patterns
Add new pattern generators in `generate_dataset.py`:

```python
def generate_custom_pattern(n_points: int = 100, **params) -> np.ndarray:
    # Your custom mathematical pattern
    t = np.linspace(0, 2*np.pi, n_points)
    x = # Custom x equation
    y = # Custom y equation
    return np.column_stack([x, y])
```

### Adjust Model Architecture
Modify model parameters in `model.py`:

```python
model = UNet1D(
    in_channels=2,
    base_channels=128,  # Increase for larger model
    channel_multipliers=(1, 2, 4, 8, 16),  # Add more levels
    attention_levels=(2, 3, 4),  # More attention
)
```

### Training Parameters
Adjust training settings in `trainer.py`:

```python
trainer = create_trainer(
    batch_size=64,
    learning_rate=5e-5,
    num_train_timesteps=2000,  # More diffusion steps
)
```

## Examples

### Quick Start
```bash
# Generate dataset
python generate_dataset.py

# Train model
python trainer.py

# Evaluate model (assuming experiment 0)
python test.py --exp_idx 0

# Generate new trajectories
python inference.py --exp_idx 0 --num_samples 10 --create_plots
```

### Advanced Usage
```bash
# Generate custom dataset
python generate_dataset.py --n_samples_per_type 2000 --n_points 200

# Train with custom parameters
python -c "
from trainer import create_trainer
trainer = create_trainer(batch_size=64, learning_rate=5e-5)
trainer.train(num_epochs=100)
"

# Generate interpolated trajectories
python inference.py --exp_idx 0 --interpolate --num_interpolations 20
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce batch size in `trainer.py`
   - Use CPU training: set `device="cpu"`

2. **Model Not Found**:
   - Check experiment index: `ls output/`
   - Ensure training completed successfully

3. **Poor Generation Quality**:
   - Train for more epochs
   - Increase model capacity
   - Adjust learning rate

### Performance Tips

1. **Faster Training**:
   - Use GPU with CUDA
   - Increase batch size if memory allows
   - Use mixed precision training

2. **Better Quality**:
   - Train for more epochs (100-200)
   - Use more diffusion timesteps (2000-4000)
   - Increase model size

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{trajectory-diffusion-tutorial,
  title={2D Trajectory Diffusion Model Tutorial},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/trajectory-diffusion-tutorial}
}
```

## Acknowledgments

- Built using PyTorch and Hugging Face Diffusers
- Inspired by DDPM paper: "Denoising Diffusion Probabilistic Models"
- UNet architecture adapted for 1D sequence modeling