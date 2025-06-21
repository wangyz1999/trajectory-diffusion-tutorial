// Interactive Trajectory Diffusion - Client-side JavaScript

class TrajectoryCanvas {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.width = this.canvas.width;
        this.height = this.canvas.height;
        this.centerX = this.width / 2;
        this.centerY = this.height / 2;
        this.scale = 200; // Scale factor for trajectory coordinates
        
        this.setupCanvas();
    }
    
    setupCanvas() {
        // Set up high DPI rendering
        const dpr = window.devicePixelRatio || 1;
        const rect = this.canvas.getBoundingClientRect();
        
        this.canvas.width = rect.width * dpr;
        this.canvas.height = rect.height * dpr;
        this.ctx.scale(dpr, dpr);
        
        this.canvas.style.width = rect.width + 'px';
        this.canvas.style.height = rect.height + 'px';
        
        this.clear();
    }
    
    clear() {
        this.ctx.clearRect(0, 0, this.width, this.height);
        
        // Draw grid
        this.drawGrid();
        
        // Draw axes
        this.drawAxes();
    }
    
    drawGrid() {
        this.ctx.strokeStyle = '#f0f0f0';
        this.ctx.lineWidth = 1;
        
        const gridSize = 50;
        
        // Vertical lines
        for (let x = 0; x <= this.width; x += gridSize) {
            this.ctx.beginPath();
            this.ctx.moveTo(x, 0);
            this.ctx.lineTo(x, this.height);
            this.ctx.stroke();
        }
        
        // Horizontal lines
        for (let y = 0; y <= this.height; y += gridSize) {
            this.ctx.beginPath();
            this.ctx.moveTo(0, y);
            this.ctx.lineTo(this.width, y);
            this.ctx.stroke();
        }
    }
    
    drawAxes() {
        this.ctx.strokeStyle = '#ddd';
        this.ctx.lineWidth = 2;
        
        // X-axis
        this.ctx.beginPath();
        this.ctx.moveTo(0, this.centerY);
        this.ctx.lineTo(this.width, this.centerY);
        this.ctx.stroke();
        
        // Y-axis
        this.ctx.beginPath();
        this.ctx.moveTo(this.centerX, 0);
        this.ctx.lineTo(this.centerX, this.height);
        this.ctx.stroke();
    }
    
    worldToCanvas(x, y) {
        // Convert world coordinates (-1.5 to 1.5) to canvas coordinates
        const canvasX = this.centerX + (x * this.scale);
        const canvasY = this.centerY - (y * this.scale); // Flip Y axis
        return { x: canvasX, y: canvasY };
    }
    
    drawTrajectory(trajectory, options = {}) {
        if (!trajectory || trajectory.length === 0) return;
        
        const {
            color = '#4169E1',
            lineWidth = 3,
            showPoints = true,
            showStartEnd = true,
            alpha = 0.8
        } = options;
        
        this.clear();
        
        // Draw trajectory line
        this.ctx.strokeStyle = color;
        this.ctx.lineWidth = lineWidth;
        this.ctx.globalAlpha = alpha;
        
        this.ctx.beginPath();
        const firstPoint = this.worldToCanvas(trajectory[0][0], trajectory[0][1]);
        this.ctx.moveTo(firstPoint.x, firstPoint.y);
        
        for (let i = 1; i < trajectory.length; i++) {
            const point = this.worldToCanvas(trajectory[i][0], trajectory[i][1]);
            this.ctx.lineTo(point.x, point.y);
        }
        this.ctx.stroke();
        
        // Draw points along trajectory
        if (showPoints) {
            this.ctx.fillStyle = color;
            for (let i = 0; i < trajectory.length; i += 5) { // Every 5th point
                const point = this.worldToCanvas(trajectory[i][0], trajectory[i][1]);
                this.ctx.beginPath();
                this.ctx.arc(point.x, point.y, 2, 0, 2 * Math.PI);
                this.ctx.fill();
            }
        }
        
        // Draw start and end markers
        if (showStartEnd && trajectory.length > 1) {
            // Start point (green)
            const startPoint = this.worldToCanvas(trajectory[0][0], trajectory[0][1]);
            this.ctx.fillStyle = '#28a745';
            this.ctx.beginPath();
            this.ctx.arc(startPoint.x, startPoint.y, 8, 0, 2 * Math.PI);
            this.ctx.fill();
            
            // End point (red)
            const endPoint = this.worldToCanvas(trajectory[trajectory.length - 1][0], trajectory[trajectory.length - 1][1]);
            this.ctx.fillStyle = '#dc3545';
            this.ctx.beginPath();
            this.ctx.arc(endPoint.x, endPoint.y, 8, 0, 2 * Math.PI);
            this.ctx.fill();
        }
        
        this.ctx.globalAlpha = 1.0;
    }
    
    drawTrajectoryData(trajectoryData, stepInfo) {
        // Draw trajectory directly from data points
        if (!trajectoryData || trajectoryData.length === 0) return;
        
        this.drawTrajectory(trajectoryData, {
            color: '#4169E1',
            lineWidth: 3,
            showPoints: false,
            showStartEnd: true,
            alpha: 0.8
        });
        
        // Add step information
        this.ctx.fillStyle = '#333';
        this.ctx.font = '16px Arial';
        this.ctx.fillText(stepInfo, 20, 30);
    }
}

class TrajectoryApp {
    constructor() {
        this.canvas = new TrajectoryCanvas('trajectory-canvas');
        this.socket = null;
        this.isGenerating = false;
        this.currentTrajectory = null;
        
        this.initializeElements();
        this.setupEventListeners();
        this.connectWebSocket();
    }
    
    initializeElements() {
        this.elements = {
            textPrompt: document.getElementById('text-prompt'),
            generateBtn: document.getElementById('generate-btn'),
            clearBtn: document.getElementById('clear-btn'),
            btnText: document.querySelector('.btn-text'),
            btnLoader: document.querySelector('.btn-loader'),
            progressContainer: document.querySelector('.progress-container'),
            progressFill: document.getElementById('progress-fill'),
            progressText: document.getElementById('progress-text'),
            statusMessage: document.getElementById('status-message'),
            connectionStatus: document.getElementById('connection-status'),
            canvasOverlay: document.querySelector('.canvas-overlay'),
            canvasTitle: document.getElementById('canvas-title'),
            canvasSubtitle: document.getElementById('canvas-subtitle'),
            canvasMainTitle: document.getElementById('canvas-main-title'),
            canvasStepInfo: document.getElementById('canvas-step-info')
        };
    }
    
    setupEventListeners() {
        // Generate button
        this.elements.generateBtn.addEventListener('click', () => {
            this.generateTrajectory();
        });
        
        // Clear button
        this.elements.clearBtn.addEventListener('click', () => {
            this.clearCanvas();
        });
        
        // Enter key in text input
        this.elements.textPrompt.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !this.isGenerating) {
                this.generateTrajectory();
            }
        });
        
        // Example buttons
        document.querySelectorAll('.example-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const prompt = btn.getAttribute('data-prompt');
                this.elements.textPrompt.value = prompt;
                this.generateTrajectory();
            });
        });
    }
    
    connectWebSocket() {
        this.socket = io();
        
        this.socket.on('connect', () => {
            console.log('Connected to server');
            this.updateConnectionStatus(true);
        });
        
        this.socket.on('disconnect', () => {
            console.log('Disconnected from server');
            this.updateConnectionStatus(false);
        });
        
        this.socket.on('connected', (data) => {
            console.log('📡 Received connected:', data);
            this.updateStatus(data.message, 'success');
        });
        
        this.socket.on('generation_started', (data) => {
            console.log('📡 Received generation_started:', data);
            this.updateStatus(`Generating: "${data.text_prompt}"`, 'info');
            this.hideCanvasOverlay();
            this.elements.canvasMainTitle.textContent = `Generating: "${data.text_prompt}"`;
            this.elements.canvasStepInfo.textContent = 'Starting diffusion process...';
        });
        
        this.socket.on('generation_step', (data) => {
            console.log('📡 Received generation_step:', `${data.step}/${data.total_steps} (${data.progress.toFixed(1)}%)`);
            this.updateProgress(data.progress);
            this.updateStatus(`Step ${data.step}/${data.total_steps} (${data.progress.toFixed(1)}%)`, 'info');
            
            // Update canvas title bar
            this.elements.canvasStepInfo.textContent = `Step ${data.step}/${data.total_steps} (${data.progress.toFixed(1)}%)`;
            
            if (data.trajectory) {
                const stepInfo = `Denoising Step ${data.step}/${data.total_steps}`;
                this.canvas.drawTrajectoryData(data.trajectory, stepInfo);
            } else {
                console.warn('⚠️ No trajectory data received');
            }
        });
        
        this.socket.on('generation_complete', (data) => {
            console.log('📡 Received generation_complete:', data);
            this.updateStatus(data.message, 'success');
            this.elements.canvasStepInfo.textContent = 'Generation completed! ✅';
            this.finishGeneration();
        });
        
        this.socket.on('error', (data) => {
            console.error('📡 Received error:', data);
            this.updateStatus(`Error: ${data.message}`, 'error');
            this.finishGeneration();
        });
    }
    
    generateTrajectory() {
        if (this.isGenerating) return;
        
        const textPrompt = this.elements.textPrompt.value.trim();
        if (!textPrompt) {
            this.updateStatus('Please enter a text description', 'warning');
            return;
        }
        
        console.log('🚀 Starting trajectory generation for:', textPrompt);
        this.startGeneration();
        
        // Send generation request (num_steps is now fixed at 1000 in backend)
        fetch('/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                text_prompt: textPrompt
            })
        })
        .then(response => response.json())
        .then(data => {
            console.log('📡 Server response:', data);
            if (data.status === 'error') {
                this.updateStatus(`Error: ${data.message}`, 'error');
                this.finishGeneration();
            } else {
                this.updateStatus('Generation started...', 'info');
            }
        })
        .catch(error => {
            console.error('❌ Network error:', error);
            this.updateStatus('Network error occurred', 'error');
            this.finishGeneration();
        });
    }
    
    startGeneration() {
        this.isGenerating = true;
        this.elements.generateBtn.disabled = true;
        this.elements.btnText.style.display = 'none';
        this.elements.btnLoader.style.display = 'inline';
        this.elements.progressContainer.style.display = 'block';
        this.updateProgress(0);
        this.updateStatus('Starting generation...', 'info');
    }
    
    finishGeneration() {
        this.isGenerating = false;
        this.elements.generateBtn.disabled = false;
        this.elements.btnText.style.display = 'inline';
        this.elements.btnLoader.style.display = 'none';
        this.elements.progressContainer.style.display = 'none';
    }
    
    clearCanvas() {
        this.canvas.clear();
        this.showCanvasOverlay();
        this.elements.canvasMainTitle.textContent = 'Trajectory Visualization';
        this.elements.canvasStepInfo.textContent = 'Enter a description and click generate';
        this.updateStatus('Canvas cleared', 'info');
    }
    
    updateProgress(percentage) {
        this.elements.progressFill.style.width = `${percentage}%`;
        this.elements.progressText.textContent = `${Math.round(percentage)}%`;
    }
    
    updateStatus(message, type = 'info') {
        this.elements.statusMessage.textContent = message;
        this.elements.statusMessage.className = `status-message status-${type}`;
        
        // Add fade-in animation
        this.elements.statusMessage.classList.remove('fade-in');
        void this.elements.statusMessage.offsetWidth; // Trigger reflow
        this.elements.statusMessage.classList.add('fade-in');
    }
    
    updateConnectionStatus(connected) {
        if (connected) {
            this.elements.connectionStatus.textContent = '🟢 Connected';
            this.elements.connectionStatus.className = 'connection-status connection-connected';
        } else {
            this.elements.connectionStatus.textContent = '🔴 Disconnected';
            this.elements.connectionStatus.className = 'connection-status connection-disconnected';
        }
    }
    
    hideCanvasOverlay() {
        this.elements.canvasOverlay.classList.add('hidden');
    }
    
    showCanvasOverlay() {
        this.elements.canvasOverlay.classList.remove('hidden');
    }
}

// Initialize the app when the page loads
document.addEventListener('DOMContentLoaded', () => {
    const app = new TrajectoryApp();
    
    // Make app globally accessible for debugging
    window.trajectoryApp = app;
    
    console.log('🎨 Interactive Trajectory Diffusion App loaded!');
}); 