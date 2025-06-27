"""
Inference Script for Multi-Agent Text-Conditioned Trajectory Diffusion with CFG
Demonstrates classifier-free guidance with different guidance scales.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple

from model import create_multi_agent_model
from dataset import MultiAgentTrajectoryDataModule, TextEncoder
from trainer import MultiAgentTrajectoryDiffusionTrainer
from diffusers import DDPMScheduler
import utils


class MultiAgentCFGInference:
    """Inference class for multi-agent trajectory generation with CFG."""
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        sequence_length: int = 100,
        max_agents: int = 5
    ):
        """
        Initialize the CFG inference system.
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device to run inference on
            sequence_length: Length of generated trajectories
            max_agents: Maximum number of agents
        """
        self.device = device
        self.sequence_length = sequence_length
        self.max_agents = max_agents
        
        # Load model configuration from checkpoint
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Create model
        self.model = create_multi_agent_model(
            max_agents=max_agents,
            sequence_length=sequence_length,
            text_emb_dim=384  # Standard text embedding dimension
        )
        
        # Load trained weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
        # Create scheduler with same config as training
        self.scheduler = DDPMScheduler(
            num_train_timesteps=checkpoint.get('scheduler_config', {}).get('num_train_timesteps', 1000),
            beta_start=checkpoint.get('scheduler_config', {}).get('beta_start', 0.0001),
            beta_end=checkpoint.get('scheduler_config', {}).get('beta_end', 0.02),
            beta_schedule=checkpoint.get('scheduler_config', {}).get('beta_schedule', 'linear'),
            clip_sample=True
        )
        
        # Create text encoder
        self.text_encoder = TextEncoder(device=device)
        
        print(f"🤖 Multi-agent CFG inference system loaded")
        print(f"📱 Device: {device}")
        print(f"🔢 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"👥 Max agents: {max_agents}")
        print(f"📏 Sequence length: {sequence_length}")
    
    def generate_trajectories(
        self,
        text_descriptions: List[str],
        agent_configurations: List[List[bool]],
        guidance_scales: List[float] = [0.0, 1.0, 4.0, 7.0, 10.0, 13.0, 15.0, 20.0],
        num_inference_steps: Optional[int] = None
    ) -> Dict[float, np.ndarray]:
        """
        Generate multi-agent trajectories with different CFG guidance scales.
        
        Args:
            text_descriptions: List of text descriptions for conditioning
            agent_configurations: List of boolean masks for active agents per sample
            guidance_scales: List of CFG guidance scales to use
            num_inference_steps: Number of denoising steps (default: use all timesteps)
            
        Returns:
            Dictionary mapping guidance scales to generated trajectories
        """
        n_samples = len(text_descriptions)
        if num_inference_steps is None:
            num_inference_steps = self.scheduler.config.num_train_timesteps
        
        # Encode text descriptions
        print("🔤 Encoding text descriptions...")
        text_embeddings = self.text_encoder.encode(text_descriptions)
        
        # Create agent masks
        agent_masks = torch.zeros(n_samples, self.max_agents, dtype=torch.bool, device=self.device)
        for i, config in enumerate(agent_configurations):
            for j, is_active in enumerate(config[:self.max_agents]):
                agent_masks[i, j] = is_active
        
        # Generate trajectories with different CFG scales
        results = {}
        
        with torch.no_grad():
            for cfg_scale in guidance_scales:
                print(f"🎯 Generating with CFG scale: {cfg_scale}")
                
                # Start from pure noise
                trajectories = torch.randn(n_samples, self.max_agents, self.sequence_length, 2).to(self.device)
                
                # Denoising loop
                timesteps = self.scheduler.timesteps
                if len(timesteps) > num_inference_steps:
                    # Use subset of timesteps for faster inference
                    step_ratio = len(timesteps) // num_inference_steps
                    timesteps = timesteps[::step_ratio]
                
                for t in tqdm(timesteps, desc=f"CFG {cfg_scale}", leave=False):
                    timestep_batch = torch.full((n_samples,), t, device=self.device).long()
                    
                    # Predict noise with CFG
                    if cfg_scale == 1.0:
                        # Standard conditional generation
                        noise_pred = self.model(trajectories, timestep_batch, text_embeddings, agent_masks)
                    else:
                        # CFG generation
                        noise_pred = self.model.forward_with_cfg(
                            trajectories, timestep_batch, text_embeddings, 
                            agent_masks, guidance_scale=cfg_scale
                        )
                    
                    # Denoising step
                    trajectories = self.scheduler.step(noise_pred, t, trajectories).prev_sample
                
                # Store results
                results[cfg_scale] = trajectories.cpu().numpy()
        
        return results
    
    def demonstrate_cfg_effects(
        self,
        sample_texts: Optional[List[str]] = None,
        sample_agent_configs: Optional[List[List[bool]]] = None,
        save_dir: str = "cfg_demo_output"
    ) -> None:
        """
        Demonstrate the effects of different CFG guidance scales.
        
        Args:
            sample_texts: Sample text descriptions (uses defaults if None)
            sample_agent_configs: Agent configurations (uses defaults if None)
            save_dir: Directory to save demonstration plots
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Use default samples if not provided
        if sample_texts is None:
            sample_texts = [
                "A circle at the center and a triangle on the right",
                "A spiral at the top left with a star at the bottom right",
                "Three agents: a heart at the center, a square on the left, and a circle on the right",
                "A figure-eight at the top and two triangles at the bottom corners"
            ]
        
        if sample_agent_configs is None:
            sample_agent_configs = [
                [True, True, False, False, False],  # 2 agents
                [True, True, False, False, False],  # 2 agents
                [True, True, True, False, False],   # 3 agents
                [True, True, True, False, False]    # 3 agents
            ]
        
        # Generate trajectories with comprehensive CFG scale range
        guidance_scales = [0.0, 1.0, 4.0, 7.0, 10.0, 13.0, 15.0, 20.0]
        results = self.generate_trajectories(
            text_descriptions=sample_texts,
            agent_configurations=sample_agent_configs,
            guidance_scales=guidance_scales,
            num_inference_steps=50  # Faster inference for demo
        )
        
        # Create agent masks for plotting
        agent_masks = np.zeros((len(sample_texts), self.max_agents), dtype=bool)
        for i, config in enumerate(sample_agent_configs):
            for j, is_active in enumerate(config[:self.max_agents]):
                agent_masks[i, j] = is_active
        
        # Plot CFG comparison with focused scales
        comparison_path = os.path.join(save_dir, 'cfg_comparison_demo.png')
        utils.plot_cfg_comparison(
            real_trajectories=np.zeros_like(results[1.0]),  # Dummy real trajectories
            generated_trajectories_dict=results,
            agent_masks=agent_masks,
            text_descriptions=sample_texts,
            guidance_scales=[0.0, 1.0, 7.0, 13.0, 20.0],  # Key scales for comparison
            epoch=0,
            save_path=comparison_path,
            n_samples=len(sample_texts)
        )
        
        # Plot CFG ablation study with all scales
        ablation_path = os.path.join(save_dir, 'cfg_ablation_demo.png')
        utils.plot_cfg_ablation_study(
            trajectories_dict=results,
            agent_masks=agent_masks,
            text_descriptions=sample_texts,
            guidance_scales=guidance_scales,
            save_path=ablation_path
        )
        
        # Analyze CFG effects
        analysis = utils.analyze_cfg_effects(
            trajectories_dict=results,
            agent_masks=agent_masks,
            guidance_scales=guidance_scales
        )
        
        # Print and save analysis
        utils.print_cfg_analysis(analysis)
        
        analysis_path = os.path.join(save_dir, 'cfg_analysis_demo.json')
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"\n✅ CFG demonstration completed!")
        print(f"📊 Results saved to: {save_dir}/")
        print(f"🎨 Plots: cfg_comparison_demo.png, cfg_ablation_demo.png")
        print(f"📈 Analysis: cfg_analysis_demo.json")
        print(f"🎯 CFG scales tested: {guidance_scales}")
    
    def interactive_generation(self) -> None:
        """Interactive text-to-trajectory generation with CFG control."""
        print("\n🎯 Interactive Multi-Agent Trajectory Generation with CFG")
        print("=" * 60)
        print("Enter text descriptions and agent configurations to generate trajectories.")
        print("Type 'quit' to exit.")
        
        while True:
            # Get text description
            text_input = input("\n📝 Enter text description: ").strip()
            if text_input.lower() in ['quit', 'exit', 'q']:
                break
            
            # Get agent configuration
            print(f"👥 Configure agents (max {self.max_agents} agents)")
            agent_config = []
            for i in range(self.max_agents):
                active = input(f"   Agent {i+1} active? (y/n): ").strip().lower().startswith('y')
                agent_config.append(active)
            
            # Get CFG scale
            try:
                cfg_scale = float(input("🎯 CFG guidance scale (0.0=unconditional, 1.0=standard, >1.0=over-guided): "))
            except ValueError:
                cfg_scale = 1.0
                print("Invalid input, using CFG scale 1.0")
            
            # Generate trajectory
            print(f"🚀 Generating trajectory with CFG scale {cfg_scale}...")
            results = self.generate_trajectories(
                text_descriptions=[text_input],
                agent_configurations=[agent_config],
                guidance_scales=[cfg_scale],
                num_inference_steps=25  # Fast inference
            )
            
            # Plot result
            trajectory = results[cfg_scale][0]  # First (and only) sample
            agent_mask = np.array(agent_config[:self.max_agents])
            
            # Quick visualization
            plt.figure(figsize=(8, 8))
            colors = plt.cm.tab10(np.linspace(0, 1, 10))
            
            for agent_idx, (traj, is_active) in enumerate(zip(trajectory, agent_mask)):
                if is_active:
                    color = colors[agent_idx % len(colors)]
                    plt.plot(traj[:, 0], traj[:, 1], color=color, linewidth=2, 
                            alpha=0.8, label=f'Agent {agent_idx+1}')
                    plt.plot(traj[0, 0], traj[0, 1], 's', color=color, markersize=8)
                    plt.plot(traj[-1, 0], traj[-1, 1], 'o', color=color, markersize=8)
            
            plt.xlim(-1.2, 1.2)
            plt.ylim(-1.2, 1.2)
            plt.grid(True, alpha=0.3)
            plt.axis('equal')
            plt.title(f'Generated Trajectory (CFG={cfg_scale})\n"{text_input}"')
            plt.legend()
            plt.show()
        
        print("👋 Interactive generation session ended.")


def main():
    """Main function to demonstrate CFG inference."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-Agent Trajectory CFG Inference")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--mode", type=str, choices=["demo", "interactive"], default="demo", 
                       help="Inference mode: demo or interactive")
    parser.add_argument("--output_dir", type=str, default="cfg_inference_output", 
                       help="Output directory for results")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    # Initialize inference system
    inference = MultiAgentCFGInference(
        model_path=args.model_path,
        device=args.device
    )
    
    if args.mode == "demo":
        # Run demonstration
        inference.demonstrate_cfg_effects(save_dir=args.output_dir)
    elif args.mode == "interactive":
        # Run interactive session
        inference.interactive_generation()


if __name__ == "__main__":
    main() 