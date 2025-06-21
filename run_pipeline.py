#!/usr/bin/env python3
"""
Complete Pipeline Runner for 2D Trajectory Diffusion Model
Runs the entire pipeline from dataset generation to inference.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("✅ SUCCESS")
        if result.stdout:
            print("Output:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("❌ FAILED")
        print("Error:", e.stderr)
        return False


def main():
    """Main pipeline runner."""
    parser = argparse.ArgumentParser(description='Run complete trajectory diffusion pipeline')
    parser.add_argument('--skip_dataset', action='store_true', help='Skip dataset generation')
    parser.add_argument('--skip_training', action='store_true', help='Skip model training')
    parser.add_argument('--skip_evaluation', action='store_true', help='Skip model evaluation')
    parser.add_argument('--skip_inference', action='store_true', help='Skip inference')
    parser.add_argument('--exp_idx', type=int, default=None, help='Use specific experiment index (for evaluation/inference)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--num_samples', type=int, default=20, help='Number of samples to generate')
    
    args = parser.parse_args()
    
    print("🚀 Starting Trajectory Diffusion Pipeline")
    print(f"Current directory: {os.getcwd()}")
    
    # Check if required files exist
    required_files = ['generate_dataset.py', 'trainer.py', 'test.py', 'inference.py']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        sys.exit(1)
    
    success_steps = []
    failed_steps = []
    
    # Step 1: Generate Dataset
    if not args.skip_dataset:
        if run_command("python generate_dataset.py", "Generating Dataset"):
            success_steps.append("Dataset Generation")
        else:
            failed_steps.append("Dataset Generation")
            print("Cannot proceed without dataset. Exiting.")
            sys.exit(1)
    else:
        print("⏭️  Skipping dataset generation")
        if not os.path.exists("data"):
            print("❌ No dataset found. Please run dataset generation first.")
            sys.exit(1)
    
    # Step 2: Train Model
    exp_idx = args.exp_idx
    if not args.skip_training:
        # Modify trainer to accept epochs parameter
        train_command = f"python -c \"from trainer import create_trainer; trainer = create_trainer(); trainer.train({args.epochs})\""
        
        if run_command(train_command, f"Training Model ({args.epochs} epochs)"):
            success_steps.append("Model Training")
            
            # Find the latest experiment if exp_idx not specified
            if exp_idx is None:
                output_dirs = list(Path("output").glob("trajectory_diffusion_*"))
                if output_dirs:
                    latest_dir = max(output_dirs, key=lambda x: int(x.name.split('_')[-1]))
                    exp_idx = int(latest_dir.name.split('_')[-1])
                    print(f"📁 Using experiment index: {exp_idx}")
        else:
            failed_steps.append("Model Training")
            print("Training failed. Skipping evaluation and inference.")
            sys.exit(1)
    else:
        print("⏭️  Skipping model training")
        if exp_idx is None:
            print("❌ No experiment index specified and training skipped.")
            print("Use --exp_idx to specify which experiment to use.")
            sys.exit(1)
    
    # Step 3: Evaluate Model
    if not args.skip_evaluation and exp_idx is not None:
        eval_command = f"python test.py --exp_idx {exp_idx}"
        
        if run_command(eval_command, f"Evaluating Model (exp_{exp_idx})"):
            success_steps.append("Model Evaluation")
        else:
            failed_steps.append("Model Evaluation")
    else:
        if exp_idx is None:
            print("⏭️  Skipping evaluation (no experiment index)")
        else:
            print("⏭️  Skipping model evaluation")
    
    # Step 4: Generate Trajectories
    if not args.skip_inference and exp_idx is not None:
        inference_command = f"python inference.py --exp_idx {exp_idx} --num_samples {args.num_samples} --create_plots --create_animations"
        
        if run_command(inference_command, f"Generating Trajectories (exp_{exp_idx})"):
            success_steps.append("Trajectory Generation")
        else:
            failed_steps.append("Trajectory Generation")
    else:
        if exp_idx is None:
            print("⏭️  Skipping inference (no experiment index)")
        else:
            print("⏭️  Skipping trajectory generation")
    
    # Summary
    print(f"\n{'='*60}")
    print("PIPELINE SUMMARY")
    print(f"{'='*60}")
    
    if success_steps:
        print("✅ Successful steps:")
        for step in success_steps:
            print(f"   - {step}")
    
    if failed_steps:
        print("❌ Failed steps:")
        for step in failed_steps:
            print(f"   - {step}")
    
    if exp_idx is not None:
        exp_dir = f"output/trajectory_diffusion_{exp_idx:03d}"
        if os.path.exists(exp_dir):
            print(f"\n📁 Results saved in: {exp_dir}")
            
            # List output files
            output_files = []
            for root, dirs, files in os.walk(exp_dir):
                for file in files:
                    if file.endswith(('.png', '.json', '.pth', '.npy', '.gif')):
                        rel_path = os.path.relpath(os.path.join(root, file), exp_dir)
                        output_files.append(rel_path)
            
            if output_files:
                print("📄 Generated files:")
                for file in sorted(output_files):
                    print(f"   - {file}")
    
    print(f"\n🎉 Pipeline completed!")
    print(f"Success rate: {len(success_steps)}/{len(success_steps) + len(failed_steps)} steps")


if __name__ == "__main__":
    main() 