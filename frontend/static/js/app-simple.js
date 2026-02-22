/**
 * Simple Safety Surveillance Dashboard
 * Clean and easy to understand
 */

class SimpleDashboard {
    constructor() {
        this.uploadedFile = null;
        this.systemReady = false;
        this.init();
    }
    
    init() {
        console.log('🚀 Dashboard starting...');
        
        // Check if system is ready
        this.checkSystem();
        
        // Setup all event listeners
        this.setupEvents();
        
        // Start polling for stats
        setInterval(() => this.updateStats(), 3000);
    }
    
    async checkSystem() {
        try {
            const response = await fetch('/api/config');
            const config = await response.json();
            
            if (config.system_ready) {
                this.systemReady = true;
                this.updateStatus('active', 'Ready');
                this.hideModal();
            } else {
                this.showModal();
            }
        } catch (error) {
            console.error('Check failed:', error);
            this.showModal();
        }
    }
    
    setupEvents() {
        // Upload area
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        
        uploadArea.addEventListener('click', () => fileInput.click());
        fileInput.addEventListener('change', (e) => this.handleFile(e.target.files[0]));
        
        // Drag and drop
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            if (e.dataTransfer.files.length > 0) {
                this.handleFile(e.dataTransfer.files[0]);
            }
        });
        
        // Buttons
        document.getElementById('analyzeBtn').addEventListener('click', () => this.analyze());
        document.getElementById('downloadBtn').addEventListener('click', () => this.download());
        document.getElementById('initBtn').addEventListener('click', () => this.initialize());
    }
    
    async initialize() {
        const ppeModel = document.getElementById('ppeModelPath').value.trim();
        const fireModel = document.getElementById('fireModelPath').value.trim();
        
        if (!ppeModel || !fireModel) {
            this.notify('Please enter both model paths', 'warning');
            return;
        }
        
        const btn = document.getElementById('initBtn');
        btn.textContent = 'Initializing...';
        btn.disabled = true;
        
        try {
            const response = await fetch('/api/initialize', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ ppe_model: ppeModel, fire_model: fireModel })
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.systemReady = true;
                this.updateStatus('active', 'Ready');
                this.hideModal();
                this.notify('System initialized successfully!', 'success');
            } else {
                this.notify(result.error || 'Initialization failed', 'error');
            }
        } catch (error) {
            this.notify('Failed to initialize', 'error');
        } finally {
            btn.textContent = 'Initialize';
            btn.disabled = false;
        }
    }
    
    async handleFile(file) {
        if (!file) return;
        
        console.log('File selected:', file.name);
        
        // Update upload area
        const uploadArea = document.getElementById('uploadArea');
        uploadArea.innerHTML = `
            <div class="upload-content">
                <div class="upload-icon">✅</div>
                <p class="upload-title">${file.name}</p>
                <p class="upload-subtitle">${this.formatSize(file.size)}</p>
                <p class="upload-formats">Click to change file</p>
            </div>
        `;
        
        // Upload to server
        const formData = new FormData();
        formData.append('file', file);
        
        try {
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.uploadedFile = result;
                document.getElementById('analyzeBtn').disabled = false;
                this.notify('File uploaded!', 'success');
            } else {
                this.notify(result.error || 'Upload failed', 'error');
            }
        } catch (error) {
            this.notify('Upload failed', 'error');
        }
    }
    
    async analyze() {
        if (!this.uploadedFile) {
            this.notify('No file uploaded', 'warning');
            return;
        }
        
        if (!this.systemReady) {
            this.notify('System not initialized', 'warning');
            this.showModal();
            return;
        }
        
        // Disable button
        document.getElementById('analyzeBtn').disabled = true;
        this.updateStatus('processing', 'Analyzing...');
        
        // Show progress
        document.getElementById('progressCard').style.display = 'block';
        
        try {
            const response = await fetch('/api/process', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    filename: this.uploadedFile.filename,
                    file_type: this.uploadedFile.file_type
                })
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.notify('Analysis started', 'info');
                this.pollProgress(result.output_filename);
            } else {
                this.notify(result.error || 'Analysis failed', 'error');
                this.resetAfterProcess();
            }
        } catch (error) {
            this.notify('Failed to start analysis', 'error');
            this.resetAfterProcess();
        }
    }
    
    async pollProgress(outputFilename) {
        const interval = setInterval(async () => {
            try {
                const response = await fetch('/api/status');
                const status = await response.json();
                
                // Update progress bar
                document.getElementById('progressFill').style.width = status.progress + '%';
                document.getElementById('progressText').textContent = status.message || 'Processing...';
                
                // Check if done
                if (!status.is_processing) {
                    clearInterval(interval);
                    
                    if (status.error) {
                        this.notify('Analysis failed: ' + status.error, 'error');
                    } else {
                        this.notify('Analysis complete!', 'success');
                        this.showResult(outputFilename);
                        this.loadAlerts();
                    }
                    
                    this.resetAfterProcess();
                }
            } catch (error) {
                console.error('Polling error:', error);
            }
        }, 1000);
    }
    
    resetAfterProcess() {
        document.getElementById('progressCard').style.display = 'none';
        document.getElementById('analyzeBtn').disabled = false;
        this.updateStatus('active', 'Ready');
        this.updateStats();
    }
    
    showResult(filename) {
        const preview = document.getElementById('previewContainer');
        const type = this.uploadedFile.file_type;
        
        if (type === 'image') {
            preview.innerHTML = `<img src="/api/preview/${filename}" alt="Result">`;
        } else {
            preview.innerHTML = `<video controls src="/api/preview/${filename}"></video>`;
        }
        
        const downloadBtn = document.getElementById('downloadBtn');
        downloadBtn.disabled = false;
        downloadBtn.dataset.filename = filename;
    }
    
    download() {
        const filename = document.getElementById('downloadBtn').dataset.filename;
        if (filename) {
            window.location.href = `/api/download/${filename}`;
            this.notify('Download started', 'success');
        }
    }
    
    async updateStats() {
        if (!this.systemReady) return;
        
        try {
            const response = await fetch('/api/stats');
            const stats = await response.json();
            
            if (!stats.error) {
                this.animateValue('statPPE', stats.ppe_violations);
                this.animateValue('statFire', stats.fire_detections);
                this.animateValue('statSmoke', stats.smoke_detections);
                this.animateValue('statAlerts', stats.total_alerts);
            }
        } catch (error) {
            console.error('Stats error:', error);
        }
    }
    
    async loadAlerts() {
        try {
            const response = await fetch('/api/alerts');
            const data = await response.json();
            
            if (data.alerts && data.alerts.length > 0) {
                document.getElementById('alertsCard').style.display = 'block';
                
                const alertsList = document.getElementById('alertsList');
                alertsList.innerHTML = data.alerts.slice(-10).reverse().map(alert => `
                    <div class="alert-item severity-${alert.severity}">
                        <div class="alert-message">${alert.message}</div>
                        <div class="alert-time">${new Date(alert.timestamp).toLocaleString()}</div>
                    </div>
                `).join('');
            }
        } catch (error) {
            console.error('Alerts error:', error);
        }
    }
    
    animateValue(id, value) {
        const element = document.getElementById(id);
        const current = parseInt(element.textContent) || 0;
        
        if (current === value) return;
        
        const duration = 500;
        const steps = 20;
        const increment = (value - current) / steps;
        let step = 0;
        
        const timer = setInterval(() => {
            step++;
            element.textContent = Math.round(current + increment * step);
            
            if (step >= steps) {
                clearInterval(timer);
                element.textContent = value;
            }
        }, duration / steps);
    }
    
    updateStatus(state, text) {
        const badge = document.getElementById('systemStatus');
        badge.className = 'status-badge ' + state;
        badge.querySelector('.status-text').textContent = text;
    }
    
    showModal() {
        document.getElementById('initModal').classList.remove('hidden');
    }
    
    hideModal() {
        document.getElementById('initModal').classList.add('hidden');
    }
    
    notify(message, type = 'info') {
        const icons = {
            success: '✅',
            error: '❌',
            warning: '⚠️',
            info: 'ℹ️'
        };
        
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.innerHTML = `
            <span class="notification-icon">${icons[type]}</span>
            <span class="notification-text">${message}</span>
        `;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.style.animation = 'slideInRight 0.3s ease reverse';
            setTimeout(() => notification.remove(), 300);
        }, 3000);
    }
    
    formatSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
    }
}

// Start the dashboard
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new SimpleDashboard();
});
