# Classifier-Free Guidance for Multi-Agent Trajectory Diffusion

This implementation includes **Classifier-Free Guidance (CFG)** for text-conditioned multi-agent trajectory generation. CFG allows fine-grained control over how strongly the model follows text conditioning during inference.

## 🎯 What is Classifier-Free Guidance?

CFG is a technique that enables controlling the strength of conditioning during generation:

- **CFG Scale 0.0**: Unconditional generation (ignores text completely)
- **CFG Scale 1.0**: Standard conditional generation (follows text normally)  
- **CFG Scale 4.0-7.0**: Moderate over-guidance (stronger text adherence)
- **CFG Scale 10.0-15.0**: Strong over-guidance (high text adherence)
- **CFG Scale 20.0**: Maximum over-guidance (strongest text adherence)

### How CFG Works

1. **Training**: Randomly drop text conditioning with probability `cfg_dropout_prob` (default 15%)
2. **Inference**: Generate both conditional and unconditional predictions
3. **Guidance**: Interpolate predictions using: `noise_pred = noise_uncond + scale * (noise_cond - noise_uncond)`

## 🚀 Training with CFG

### Basic Training
```python
from trainer import create_multi_agent_trainer

# Create trainer with CFG enabled
trainer = create_multi_agent_trainer(
    data_dir="data",
    batch_size=64,
    max_agents=5,
    cfg_dropout_prob=0.15,  # 15% chance to drop text conditioning
    exp_name="multi_agent_cfg"
)

# Train with CFG
trainer.train(num_epochs=50)
```

### CFG Training Features

- **Automatic CFG Dropout**: Text embeddings randomly set to zero during training
- **Progress Monitoring**: CFG comparison plots generated during training
- **Validation**: Works with existing validation pipeline
- **Checkpointing**: CFG parameters saved in model checkpoints

## 🎨 Inference with CFG

### Using the Inference Script
```bash
# Run CFG demonstration
python inference.py --model_path best_model.pth --mode demo

# Interactive CFG generation
python inference.py --model_path best_model.pth --mode interactive
```

### Programmatic Inference
```python
from inference import MultiAgentCFGInference

# Load trained model
inference = MultiAgentCFGInference("path/to/model.pth")

# Generate with different CFG scales
results = inference.generate_trajectories(
    text_descriptions=[
        "A circle at the center and a triangle on the right",
        "A spiral at the top with a star at the bottom"
    ],
    agent_configurations=[
        [True, True, False, False, False],  # 2 agents
        [True, True, False, False, False]   # 2 agents
    ],
    guidance_scales=[0.0, 1.0, 7.0, 13.0, 20.0]  # Comprehensive range
)
```

## 📊 CFG Analysis and Visualization

### Automatic Plots During Training

1. **Training Progress**: Standard real vs generated comparisons
2. **CFG Comparison**: Side-by-side comparison of different guidance scales
3. **Final Analysis**: Comprehensive CFG ablation study at end of training

### Plot Types Generated

- `cfg_comparison_epoch_XXX.png`: Compare key CFG scales across samples
- `final_cfg_comprehensive_comparison.png`: Final CFG comparison
- `final_cfg_ablation_study.png`: Single sample across all CFG scales (0-20)
- `cfg_analysis.json`: Quantitative analysis of CFG effects

### CFG Effects Analysis

The system automatically analyzes how CFG affects:
- **Path Length**: Total trajectory length
- **Displacement**: Start-to-end distance  
- **Position Spread**: Spatial variance
- **Trajectory Structure**: Adherence to text descriptions

## 🔧 Configuration Options

### CFG Training Parameters

```python
trainer = create_multi_agent_trainer(
    cfg_dropout_prob=0.15,     # Probability to drop text conditioning
    # Other standard parameters...
)
```

### CFG Inference Parameters

```python
results = inference.generate_trajectories(
    guidance_scales=[0.0, 1.0, 4.0, 7.0, 10.0, 13.0, 15.0, 20.0],  # Full CFG range
    num_inference_steps=50,                                          # Denoising steps
    # Other parameters...
)
```

## 📈 Expected CFG Behavior

### CFG Scale Effects

- **0.0 (Unconditional)**: 
  - Ignores text completely
  - More random/diverse trajectories
  - May not follow spatial descriptions

- **1.0 (Standard)**:
  - Balanced conditioning
  - Good text adherence
  - Natural trajectory diversity

- **4.0-7.0 (Moderate Over-guidance)**:
  - Increased text adherence
  - More structured trajectories
  - Good balance of control and diversity

- **10.0-15.0 (Strong Over-guidance)**:
  - Strong text adherence
  - Highly structured trajectories
  - Reduced diversity, increased precision

- **20.0 (Maximum Over-guidance)**:
  - Maximum text adherence
  - Very structured/deterministic trajectories
  - Potential for over-fitting to text

### Use Cases by CFG Scale

- **Creative Exploration**: CFG 0.0-1.0
- **Balanced Generation**: CFG 4.0-7.0
- **Precise Control**: CFG 10.0-13.0  
- **Maximum Adherence**: CFG 15.0-20.0

## 🎯 Best Practices

### Training
- Use `cfg_dropout_prob=0.10-0.20` for good balance
- Monitor CFG comparison plots during training
- Ensure model learns both conditional and unconditional generation

### Inference  
- Start with CFG 1.0 for baseline
- Experiment with CFG 7.0-13.0 for optimal control
- Use CFG 0.0 to verify unconditional capability
- Try CFG 20.0 for maximum text adherence

### Troubleshooting
- If CFG 0.0 looks similar to CFG 1.0: Increase `cfg_dropout_prob`
- If high CFG scales produce artifacts: Check training convergence
- If text adherence is poor: Try higher CFG scales (10.0-20.0)
- If trajectories become too rigid: Lower CFG scale (4.0-7.0)

## 📚 Technical Details

### Model Architecture Changes
- Added `text_null_embedding` parameter for unconditional generation
- Modified `forward()` to handle `None` text embeddings
- Added `forward_with_cfg()` method for efficient CFG inference

### Training Loop Changes  
- Random CFG dropout applied per batch
- CFG statistics tracked and logged
- Multi-scale validation during training

### Plotting Enhancements
- Large subplot layouts for better visualization
- Solid lines for all trajectory plots
- Multi-line text display without truncation
- Color-coded CFG scale visualization
- Quantitative CFG effects analysis

## 🔗 Related Files

- `model.py`: CFG-enabled model architecture
- `trainer.py`: CFG training implementation  
- `inference.py`: CFG inference and demonstration
- `utils.py`: CFG plotting and analysis utilities

## 💡 Tips for Success

1. **Start Simple**: Begin with CFG 1.0 baseline
2. **Monitor Training**: Watch CFG comparison plots for quality
3. **Experiment Systematically**: Test CFG 0, 1, 7, 13, 20 for different use cases
4. **Analyze Results**: Use quantitative metrics to compare CFG effects
5. **Iterate**: Adjust CFG dropout probability based on results
6. **Scale Appropriately**: Use higher CFG (10-20) for precise control, lower (1-7) for creativity 