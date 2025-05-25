/**
 * Voice Manager for LexAI Web Client
 * Handles voice cloning, voice library management, and TTS voice selection
 */

class VoiceManager {
    constructor(apiBaseUrl) {
        this.apiBaseUrl = apiBaseUrl;
        this.voices = [];
        this.uploadInProgress = false;
        
        // UI Elements (will be bound later)
        this.elements = {};
        
        this.init();
    }
    
    async init() {
        this.bindElements();
        this.setupEventListeners();
        await this.loadVoices();
        console.log('Voice manager initialized');
    }
    
    bindElements() {
        const elementIds = [
            'voiceUploadArea', 'voiceFileInput', 'voiceName', 'cloneVoiceBtn',
            'uploadProgress', 'progressFill', 'progressText', 'voiceList'
        ];
        
        elementIds.forEach(id => {
            this.elements[id] = document.getElementById(id);
        });
    }
    
    setupEventListeners() {
        // File upload area
        this.elements.voiceUploadArea.addEventListener('click', () => {
            this.elements.voiceFileInput.click();
        });
        
        this.elements.voiceUploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.elements.voiceUploadArea.classList.add('drag-over');
        });
        
        this.elements.voiceUploadArea.addEventListener('dragleave', () => {
            this.elements.voiceUploadArea.classList.remove('drag-over');
        });
        
        this.elements.voiceUploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            this.elements.voiceUploadArea.classList.remove('drag-over');
            this.handleFilesDrop(e.dataTransfer.files);
        });
        
        // File input change
        this.elements.voiceFileInput.addEventListener('change', (e) => {
            this.handleFilesDrop(e.target.files);
        });
        
        // Clone voice button
        this.elements.cloneVoiceBtn.addEventListener('click', () => {
            this.cloneVoice();
        });
    }
    
    async loadVoices() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/api/tts/voices`);
            if (!response.ok) {
                throw new Error(`Failed to load voices: ${response.statusText}`);
            }
            
            const data = await response.json();
            this.voices = data.voices || [];
            this.updateVoiceList();
            
        } catch (error) {
            console.error('Failed to load voices:', error);
            this.showError('Failed to load voice library');
        }
    }
    
    updateVoiceList() {
        const voiceList = this.elements.voiceList;
        voiceList.innerHTML = '';
        
        if (this.voices.length === 0) {
            voiceList.innerHTML = `
                <div class="empty-voice-list">
                    <i class="fas fa-microphone-slash"></i>
                    <p>No custom voices available. Upload audio samples to create your first voice.</p>
                </div>
            `;
            return;
        }
        
        this.voices.forEach(voice => {
            const voiceItem = this.createVoiceItem(voice);
            voiceList.appendChild(voiceItem);
        });
    }
    
    createVoiceItem(voice) {
        const voiceItem = document.createElement('div');
        voiceItem.className = 'voice-item';
        voiceItem.innerHTML = `
            <div class="voice-info">
                <div class="voice-avatar">
                    <i class="fas fa-user"></i>
                </div>
                <div class="voice-details">
                    <h4>${voice.name}</h4>
                    <p>Created: ${new Date(voice.created_at).toLocaleDateString()}</p>
                    <p>Language: ${voice.language || 'auto-detect'}</p>
                    ${voice.sample_count ? `<p>Samples: ${voice.sample_count}</p>` : ''}
                </div>
            </div>
            <div class="voice-actions">
                <button class="btn btn-small btn-outline" onclick="voiceManager.testVoice('${voice.id}')">
                    <i class="fas fa-play"></i> Test
                </button>
                <button class="btn btn-small btn-outline" onclick="voiceManager.downloadVoice('${voice.id}')">
                    <i class="fas fa-download"></i>
                </button>
                <button class="btn btn-small btn-danger" onclick="voiceManager.deleteVoice('${voice.id}')">
                    <i class="fas fa-trash"></i>
                </button>
            </div>
        `;
        
        return voiceItem;
    }
    
    handleFilesDrop(files) {
        if (this.uploadInProgress) {
            this.showError('Upload already in progress');
            return;
        }
        
        const audioFiles = Array.from(files).filter(file => 
            file.type.startsWith('audio/') || 
            file.name.toLowerCase().match(/\.(mp3|wav|ogg|m4a|flac|aac)$/)
        );
        
        if (audioFiles.length === 0) {
            this.showError('Please select audio files (MP3, WAV, OGG, M4A, FLAC, AAC)');
            return;
        }
        
        // Update UI to show selected files
        this.elements.voiceUploadArea.innerHTML = `
            <i class="fas fa-file-audio"></i>
            <p>${audioFiles.length} audio file(s) selected</p>
            <div class="selected-files">
                ${audioFiles.map(file => `<div class="file-item">${file.name} (${this.formatFileSize(file.size)})</div>`).join('')}
            </div>
        `;
        
        // Store files for upload
        this.selectedFiles = audioFiles;
    }
    
    async cloneVoice() {
        if (!this.selectedFiles || this.selectedFiles.length === 0) {
            this.showError('Please select audio files first');
            return;
        }
        
        const voiceName = this.elements.voiceName.value.trim();
        if (!voiceName) {
            this.showError('Please enter a name for the voice');
            return;
        }
        
        if (this.uploadInProgress) {
            this.showError('Upload already in progress');
            return;
        }
        
        try {
            this.uploadInProgress = true;
            this.showProgress(0);
            this.elements.cloneVoiceBtn.disabled = true;
            
            // Create FormData for upload
            const formData = new FormData();
            formData.append('voice_name', voiceName);
            formData.append('language', 'auto-detect'); // Let the system detect language
            
            // Add all audio files
            this.selectedFiles.forEach((file, index) => {
                formData.append('audio_files', file);
            });
            
            // Upload with progress tracking
            const response = await this.uploadWithProgress(
                `${this.apiBaseUrl}/api/tts/voices/clone`,
                formData
            );
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Voice cloning failed');
            }
            
            const result = await response.json();
            this.showSuccess(`Voice "${voiceName}" cloned successfully!`);
            
            // Reset form
            this.resetUploadForm();
            
            // Reload voices
            await this.loadVoices();
            
            // Update main voice selector
            if (window.lexaiApp) {
                await window.lexaiApp.loadVoices();
            }
            
        } catch (error) {
            console.error('Voice cloning failed:', error);
            this.showError('Voice cloning failed: ' + error.message);
        } finally {
            this.uploadInProgress = false;
            this.elements.cloneVoiceBtn.disabled = false;
            this.hideProgress();
        }
    }
    
    async uploadWithProgress(url, formData) {
        return new Promise((resolve, reject) => {
            const xhr = new XMLHttpRequest();
            
            xhr.upload.addEventListener('progress', (e) => {
                if (e.lengthComputable) {
                    const progress = (e.loaded / e.total) * 100;
                    this.showProgress(progress);
                }
            });
            
            xhr.addEventListener('load', () => {
                resolve({
                    ok: xhr.status >= 200 && xhr.status < 300,
                    status: xhr.status,
                    statusText: xhr.statusText,
                    json: () => Promise.resolve(JSON.parse(xhr.responseText))
                });
            });
            
            xhr.addEventListener('error', () => {
                reject(new Error('Upload failed'));
            });
            
            xhr.open('POST', url);
            xhr.send(formData);
        });
    }
    
    async testVoice(voiceId) {
        try {
            const testText = "Hello! This is a test of your cloned voice. How does it sound?";
            
            const response = await fetch(`${this.apiBaseUrl}/api/tts/synthesize`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    text: testText,
                    voice_id: voiceId,
                    language: 'en'
                })
            });
            
            if (!response.ok) {
                throw new Error('Failed to generate test audio');
            }
            
            // Get audio blob and play it
            const audioBlob = await response.blob();
            const audioUrl = URL.createObjectURL(audioBlob);
            
            const audio = new Audio(audioUrl);
            audio.play();
            
            // Clean up URL after playing
            audio.addEventListener('ended', () => {
                URL.revokeObjectURL(audioUrl);
            });
            
            this.showSuccess('Playing voice test...');
            
        } catch (error) {
            console.error('Voice test failed:', error);
            this.showError('Failed to test voice: ' + error.message);
        }
    }
    
    async downloadVoice(voiceId) {
        try {
            const response = await fetch(`${this.apiBaseUrl}/api/tts/voices/${voiceId}/export`);
            
            if (!response.ok) {
                throw new Error('Failed to export voice');
            }
            
            const blob = await response.blob();
            const url = URL.createObjectURL(blob);
            
            const a = document.createElement('a');
            a.href = url;
            a.download = `voice-${voiceId}.zip`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            
            URL.revokeObjectURL(url);
            this.showSuccess('Voice exported successfully');
            
        } catch (error) {
            console.error('Voice export failed:', error);
            this.showError('Failed to export voice: ' + error.message);
        }
    }
    
    async deleteVoice(voiceId) {
        if (!confirm('Are you sure you want to delete this voice? This action cannot be undone.')) {
            return;
        }
        
        try {
            const response = await fetch(`${this.apiBaseUrl}/api/tts/voices/${voiceId}`, {
                method: 'DELETE'
            });
            
            if (!response.ok) {
                throw new Error('Failed to delete voice');
            }
            
            this.showSuccess('Voice deleted successfully');
            
            // Reload voices
            await this.loadVoices();
            
            // Update main voice selector
            if (window.lexaiApp) {
                await window.lexaiApp.loadVoices();
            }
            
        } catch (error) {
            console.error('Voice deletion failed:', error);
            this.showError('Failed to delete voice: ' + error.message);
        }
    }
    
    resetUploadForm() {
        this.elements.voiceName.value = '';
        this.elements.voiceFileInput.value = '';
        this.selectedFiles = null;
        
        this.elements.voiceUploadArea.innerHTML = `
            <i class="fas fa-cloud-upload-alt"></i>
            <p>Drop audio files here or click to upload</p>
        `;
    }
    
    showProgress(percentage) {
        this.elements.uploadProgress.style.display = 'block';
        this.elements.progressFill.style.width = `${percentage}%`;
        this.elements.progressText.textContent = `${Math.round(percentage)}%`;
    }
    
    hideProgress() {
        this.elements.uploadProgress.style.display = 'none';
    }
    
    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
    
    showError(message) {
        if (window.lexaiApp) {
            window.lexaiApp.showNotification(message, 'error');
        } else {
            alert('Error: ' + message);
        }
    }
    
    showSuccess(message) {
        if (window.lexaiApp) {
            window.lexaiApp.showNotification(message, 'success');
        } else {
            alert('Success: ' + message);
        }
    }
    
    // Get voices for external use
    getVoices() {
        return this.voices;
    }
    
    // Find voice by ID
    getVoiceById(voiceId) {
        return this.voices.find(voice => voice.id === voiceId);
    }
    
    // Refresh voice list (called from main app)
    async refreshVoiceList() {
        await this.loadVoices();
    }
    
    // Import voice from file
    async importVoice(file) {
        try {
            const formData = new FormData();
            formData.append('voice_file', file);
            
            const response = await fetch(`${this.apiBaseUrl}/api/tts/voices/import`, {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Voice import failed');
            }
            
            const result = await response.json();
            this.showSuccess(`Voice "${result.name}" imported successfully!`);
            
            // Reload voices
            await this.loadVoices();
            
            return result;
            
        } catch (error) {
            console.error('Voice import failed:', error);
            this.showError('Voice import failed: ' + error.message);
            throw error;
        }
    }
    
    // Get voice statistics
    async getVoiceStats(voiceId) {
        try {
            const response = await fetch(`${this.apiBaseUrl}/api/tts/voices/${voiceId}/stats`);
            
            if (!response.ok) {
                throw new Error('Failed to get voice statistics');
            }
            
            return await response.json();
            
        } catch (error) {
            console.error('Failed to get voice stats:', error);
            throw error;
        }
    }
    
    // Validate audio file for voice cloning
    validateAudioFile(file) {
        const maxSize = 50 * 1024 * 1024; // 50MB
        const minSize = 10 * 1024; // 10KB
        const supportedTypes = [
            'audio/mpeg', 'audio/wav', 'audio/ogg', 
            'audio/mp4', 'audio/flac', 'audio/aac'
        ];
        
        const errors = [];
        
        if (file.size > maxSize) {
            errors.push(`File too large: ${this.formatFileSize(file.size)} (max: ${this.formatFileSize(maxSize)})`);
        }
        
        if (file.size < minSize) {
            errors.push(`File too small: ${this.formatFileSize(file.size)} (min: ${this.formatFileSize(minSize)})`);
        }
        
        if (!supportedTypes.includes(file.type) && !file.name.toLowerCase().match(/\.(mp3|wav|ogg|m4a|flac|aac)$/)) {
            errors.push('Unsupported file format. Please use MP3, WAV, OGG, M4A, FLAC, or AAC');
        }
        
        return {
            valid: errors.length === 0,
            errors: errors
        };
    }
    
    // Get recommended voice cloning guidelines
    getVoiceGuidelines() {
        return {
            duration: {
                min: 10, // seconds
                recommended: 30,
                max: 300
            },
            quality: {
                sampleRate: 'At least 16kHz',
                bitRate: 'At least 128kbps',
                format: 'WAV or high-quality MP3'
            },
            content: {
                speech: 'Clear, natural speech',
                background: 'Minimal background noise',
                emotion: 'Neutral or expressive tone',
                variety: 'Different sentences and sounds'
            },
            tips: [
                'Record in a quiet environment',
                'Use a good quality microphone',
                'Speak clearly and naturally',
                'Include various phonemes and sounds',
                'Avoid long pauses or silence',
                'Multiple short clips work better than one long clip'
            ]
        };
    }
}