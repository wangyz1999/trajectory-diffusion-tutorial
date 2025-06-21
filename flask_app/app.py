"""
Interactive Trajectory Diffusion Flask Web Application
Real-time text-conditioned trajectory generation with canvas visualization.
"""

import os
import sys
import json
import numpy as np
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import torch
from threading import Thread
import time

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference import TrajectoryDiffusionInference
from dataset import TextEncoder
import utils

app = Flask(__name__)
app.config['SECRET_KEY'] = 'trajectory_diffusion_secret'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global variables
inferencer = None
text_encoder = None
current_session_id = None

def initialize_model():
    """Initialize the diffusion model and text encoder."""
    global inferencer, text_encoder
    
    try:
        # Find the latest experiment directory
        exp_dir = utils.find_experiment_dir(0, "test_plotting", "../output")
        print(f"🚀 Loading model from: {exp_dir}")
        
        # Initialize inference
        inferencer = TrajectoryDiffusionInference(exp_dir, device="cuda")
        text_encoder = TextEncoder(device="cuda")
        
        print("✅ Model initialized successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error initializing model: {e}")
        import traceback
        traceback.print_exc()
        return False

@app.route('/')
def index():
    """Main page with the interactive interface."""
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_trajectory():
    """Generate trajectory and return animation frames."""
    try:
        data = request.get_json()
        text_prompt = data.get('text_prompt', 'A perfect circle')
        # Always use 1000 steps for high quality
        num_steps = 1000
        
        print(f"🎨 Generating trajectory for: '{text_prompt}'")
        
        # Start generation in a separate thread to avoid blocking
        thread = Thread(target=generate_with_steps, args=(text_prompt, num_steps, current_session_id))
        thread.daemon = True  # Make thread daemon so it doesn't block app shutdown
        thread.start()
        
        return jsonify({'status': 'started', 'message': 'Generation started'})
        
    except Exception as e:
        print(f"❌ Error in generation: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)})

def generate_with_steps(text_prompt, num_steps, session_id):
    """Generate trajectory with intermediate steps and emit to client."""
    try:
        if inferencer is None:
            socketio.emit('error', {'message': 'Model not initialized'}, room=session_id)
            return
        
        # Encode text
        text_embedding = text_encoder.encode([text_prompt])
        text_embedding = text_embedding.to(inferencer.device)
        
        # Get sequence length from config
        seq_len = inferencer.config['data_params']['sequence_length']
        
        # Start from pure noise
        trajectory = torch.randn(1, 2, seq_len).to(inferencer.device)
        
        # Use all 1000 timesteps but only show every 30th frame for animation
        total_steps = inferencer.scheduler.config.num_train_timesteps
        steps = list(reversed(range(total_steps)))
        
        # Emit initial state
        socketio.emit('generation_started', {
            'total_steps': len(steps),
            'text_prompt': text_prompt
        }, room=session_id)
        
        with torch.no_grad():
            for i, t in enumerate(steps):
                timesteps = torch.full((1,), t, device=inferencer.device).long()
                
                # Predict noise
                noise_pred = inferencer.model(trajectory, timesteps, text_embedding)
                
                # Remove predicted noise
                trajectory = inferencer.scheduler.step(noise_pred, t, trajectory).prev_sample
                
                # Only emit updates every 30 steps or on the last step for performance
                if i % 30 == 0 or i == len(steps) - 1:
                    # Convert to numpy for visualization
                    traj_np = trajectory.cpu().numpy().transpose(0, 2, 1)[0]  # [seq_len, 2]
                    
                    # Send trajectory data directly (no matplotlib)
                    progress_data = {
                        'step': i + 1,
                        'total_steps': len(steps),
                        'trajectory': traj_np.tolist(),  # Send raw trajectory data
                        'progress': (i + 1) / len(steps) * 100
                    }
                    socketio.emit('generation_step', progress_data, room=session_id)
                    
                    # Small delay to make animation visible
                    time.sleep(0.05)
        
        # Emit completion
        socketio.emit('generation_complete', {
            'message': 'Generation completed successfully!'
        }, room=session_id)
        
    except Exception as e:
        print(f"❌ Error in generation: {e}")
        import traceback
        traceback.print_exc()
        socketio.emit('error', {'message': str(e)}, room=session_id)

# Removed matplotlib plotting function - now drawing directly on canvas

@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    global current_session_id
    current_session_id = request.sid
    print(f'🔌 Client connected with session ID: {current_session_id}')
    emit('connected', {'message': 'Connected to trajectory diffusion server'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    print(f'🔌 Client disconnected: {request.sid}')

if __name__ == '__main__':
    print("🚀 Starting Trajectory Diffusion Flask App...")
    
    # Initialize model
    if initialize_model():
        print("🌐 Starting web server...")
        socketio.run(app, debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to initialize model. Exiting.") 