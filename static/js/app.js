/**
 * LexAI Web Client - Main Application
 * Handles WebSocket connections, UI management, and conversation flow
 */

class LexAIApp {
    constructor() {
        this.websocket = null;
        this.sessionId = null;
        this.isRecording = false;
        this.isConnected = false;
        this.audioHandler = null;
        this.voiceManager = null;
        this.conversation = [];
        
        // Configuration
        this.config = {
            // Determine if we're running on localhost or external server
            wsProtocol: window.location.protocol === 'https:' ? 'wss:' : 'ws:',
            host: window.location.hostname,
            port: window.location.port || '8000',
            apiBaseUrl: window.location.origin,
            reconnectAttempts: 0,
            maxReconnectAttempts: 5,
            reconnectDelay: 2000,
            heartbeatInterval: 30000
        };
        
        // UI Elements
        this.elements = {};
        
        this.init();
    }
    
    async init() {
        this.bindElements();
        this.setupEventListeners();
        this.setupSettings();
        
        // Initialize audio handler
        try {
            this.audioHandler = new AudioHandler();
            await this.audioHandler.init();
        } catch (error) {
            this.showNotification('Failed to initialize audio: ' + error.message, 'error');
            console.error('Audio initialization failed:', error);
        }
        
        // Initialize voice manager
        this.voiceManager = new VoiceManager(this.config.apiBaseUrl);
        await this.voiceManager.init();
        
        // Load available voices
        await this.loadVoices();
        
        // Auto-connect
        this.connect();
        
        console.log('LexAI app initialized');
    }
    
    bindElements() {
        const elementIds = [
            'connectionStatus', 'statusIndicator', 'statusText',
            'newSessionBtn', 'clearConversationBtn', 'sessionId',
            'voiceSelect', 'languageSelect', 'recordBtn', 'audioVisualizer',
            'recordingStatus', 'conversationMessages', 'typingIndicator',
            'exportConversationBtn', 'voiceModal', 'voiceModalClose',
            'voiceManagementBtn', 'settingsPanel', 'settingsToggle',
            'settingsContent', 'autoPlayToggle', 'timestampsToggle',
            'audioQualitySelect', 'loadingOverlay', 'loadingText',
            'notificationContainer', 'audioPlayer'
        ];
        
        elementIds.forEach(id => {
            this.elements[id] = document.getElementById(id);
        });
    }
    
    setupEventListeners() {
        // Session controls
        this.elements.newSessionBtn.addEventListener('click', () => this.createNewSession());
        this.elements.clearConversationBtn.addEventListener('click', () => this.clearConversation());
        
        // Recording controls
        this.elements.recordBtn.addEventListener('click', () => this.toggleRecording());
        
        // Voice and language selection
        this.elements.voiceSelect.addEventListener('change', (e) => this.setVoice(e.target.value));
        this.elements.languageSelect.addEventListener('change', (e) => this.setLanguage(e.target.value));
        
        // Settings
        this.elements.settingsToggle.addEventListener('click', () => this.toggleSettings());
        this.elements.voiceManagementBtn.addEventListener('click', () => this.openVoiceModal());
        this.elements.voiceModalClose.addEventListener('click', () => this.closeVoiceModal());
        
        // Export conversation
        this.elements.exportConversationBtn.addEventListener('click', () => this.exportConversation());
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => this.handleKeyboardShortcuts(e));
        
        // Window events
        window.addEventListener('beforeunload', () => this.disconnect());
        window.addEventListener('online', () => this.handleOnline());
        window.addEventListener('offline', () => this.handleOffline());
    }
    
    setupSettings() {
        // Load settings from localStorage
        const settings = JSON.parse(localStorage.getItem('lexai-settings') || '{}');
        
        if (settings.autoPlay !== undefined) {
            this.elements.autoPlayToggle.checked = settings.autoPlay;
        }
        
        if (settings.showTimestamps !== undefined) {
            this.elements.timestampsToggle.checked = settings.showTimestamps;
        }
        
        if (settings.audioQuality) {
            this.elements.audioQualitySelect.value = settings.audioQuality;
        }
        
        // Setup setting change listeners
        this.elements.autoPlayToggle.addEventListener('change', () => this.saveSettings());
        this.elements.timestampsToggle.addEventListener('change', () => this.saveSettings());
        this.elements.audioQualitySelect.addEventListener('change', () => this.saveSettings());
    }
    
    saveSettings() {
        const settings = {
            autoPlay: this.elements.autoPlayToggle.checked,
            showTimestamps: this.elements.timestampsToggle.checked,
            audioQuality: this.elements.audioQualitySelect.value
        };
        
        localStorage.setItem('lexai-settings', JSON.stringify(settings));
    }
    
    async connect() {
        if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
            return;
        }
        
        this.showLoading('Connecting to LexAI...');
        
        try {
            const wsUrl = `${this.config.wsProtocol}//${this.config.host}:${this.config.port}/ws/audio/${this.sessionId || 'new'}`;
            console.log('Connecting to:', wsUrl);
            
            this.websocket = new WebSocket(wsUrl);
            this.setupWebSocketHandlers();
            
        } catch (error) {
            console.error('WebSocket connection failed:', error);
            this.handleConnectionError(error);
        }
    }
    
    setupWebSocketHandlers() {
        this.websocket.onopen = (event) => {
            console.log('WebSocket connected');
            this.isConnected = true;
            this.config.reconnectAttempts = 0;
            this.updateConnectionStatus('connected', 'Connected');
            this.hideLoading();
            this.showNotification('Connected to LexAI', 'success');
            
            // Start heartbeat
            this.startHeartbeat();
        };
        
        this.websocket.onmessage = async (event) => {
            try {
                if (event.data instanceof Blob) {
                    // Audio response from server
                    await this.handleAudioResponse(event.data);
                } else {
                    // JSON message from server
                    const message = JSON.parse(event.data);
                    await this.handleServerMessage(message);
                }
            } catch (error) {
                console.error('Error handling WebSocket message:', error);
            }
        };
        
        this.websocket.onclose = (event) => {
            console.log('WebSocket disconnected:', event.code, event.reason);
            this.isConnected = false;
            this.updateConnectionStatus('disconnected', 'Disconnected');
            this.stopHeartbeat();
            
            if (!event.wasClean && this.config.reconnectAttempts < this.config.maxReconnectAttempts) {
                this.scheduleReconnect();
            }
        };
        
        this.websocket.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.handleConnectionError(error);
        };
    }
    
    async handleServerMessage(message) {
        console.log('Received message:', message);
        
        switch (message.type) {
            case 'session_created':
                this.sessionId = message.session_id;
                this.elements.sessionId.textContent = this.sessionId;
                break;
                
            case 'transcription':
                this.addMessage('user', message.text, {
                    confidence: message.confidence,
                    language: message.language
                });
                break;
                
            case 'response':
                this.hideTypingIndicator();
                this.addMessage('assistant', message.text, {
                    model: message.model,
                    processingTime: message.processing_time_ms
                });
                break;
                
            case 'error':
                this.showNotification(message.error, 'error');
                this.hideTypingIndicator();
                break;
                
            case 'audio_chunk':
                // Audio will be handled by the binary message handler
                break;
                
            case 'processing':
                this.showTypingIndicator();
                break;
                
            default:
                console.log('Unknown message type:', message.type);
        }
    }
    
    async handleAudioResponse(audioBlob) {
        if (this.elements.autoPlayToggle.checked) {
            try {
                const audioUrl = URL.createObjectURL(audioBlob);
                this.elements.audioPlayer.src = audioUrl;
                await this.elements.audioPlayer.play();
                
                // Clean up object URL after playing
                this.elements.audioPlayer.addEventListener('ended', () => {
                    URL.revokeObjectURL(audioUrl);
                }, { once: true });
                
            } catch (error) {
                console.error('Error playing audio response:', error);
                this.showNotification('Failed to play audio response', 'error');
            }
        }
    }
    
    scheduleReconnect() {
        this.config.reconnectAttempts++;
        const delay = this.config.reconnectDelay * Math.pow(2, this.config.reconnectAttempts - 1);
        
        this.updateConnectionStatus('reconnecting', `Reconnecting in ${delay/1000}s... (${this.config.reconnectAttempts}/${this.config.maxReconnectAttempts})`);
        
        setTimeout(() => {
            if (!this.isConnected) {
                this.connect();
            }
        }, delay);
    }
    
    startHeartbeat() {
        this.heartbeatInterval = setInterval(() => {
            if (this.isConnected && this.websocket.readyState === WebSocket.OPEN) {
                this.websocket.send(JSON.stringify({ type: 'ping' }));
            }
        }, this.config.heartbeatInterval);
    }
    
    stopHeartbeat() {
        if (this.heartbeatInterval) {
            clearInterval(this.heartbeatInterval);
            this.heartbeatInterval = null;
        }
    }
    
    async toggleRecording() {
        if (!this.isConnected) {
            this.showNotification('Please connect to the server first', 'warning');
            return;
        }
        
        if (!this.audioHandler) {
            this.showNotification('Audio handler not initialized', 'error');
            return;
        }
        
        if (this.isRecording) {
            await this.stopRecording();
        } else {
            await this.startRecording();
        }
    }
    
    async startRecording() {
        try {
            await this.audioHandler.startRecording((audioData) => {
                // Send audio chunks to server
                if (this.isConnected && this.websocket.readyState === WebSocket.OPEN) {
                    this.websocket.send(audioData);
                }
            });
            
            this.isRecording = true;
            this.updateRecordingUI(true);
            this.elements.recordingStatus.textContent = 'Recording... Speak now';
            
        } catch (error) {
            console.error('Failed to start recording:', error);
            this.showNotification('Failed to start recording: ' + error.message, 'error');
        }
    }
    
    async stopRecording() {
        try {
            await this.audioHandler.stopRecording();
            this.isRecording = false;
            this.updateRecordingUI(false);
            this.elements.recordingStatus.textContent = 'Processing...';
            
            // Send end-of-audio signal
            if (this.isConnected && this.websocket.readyState === WebSocket.OPEN) {
                this.websocket.send(JSON.stringify({ type: 'audio_end' }));
            }
            
        } catch (error) {
            console.error('Failed to stop recording:', error);
            this.showNotification('Failed to stop recording: ' + error.message, 'error');
        }
    }
    
    updateRecordingUI(isRecording) {
        const recordBtn = this.elements.recordBtn;
        const recordIcon = recordBtn.querySelector('i');
        const recordText = recordBtn.querySelector('span');
        
        if (isRecording) {
            recordBtn.classList.add('recording');
            recordIcon.className = 'fas fa-stop';
            recordText.textContent = 'Stop Recording';
            this.startAudioVisualization();
        } else {
            recordBtn.classList.remove('recording');
            recordIcon.className = 'fas fa-microphone';
            recordText.textContent = 'Start Recording';
            this.stopAudioVisualization();
        }
    }
    
    startAudioVisualization() {
        const bars = this.elements.audioVisualizer.querySelectorAll('.bar');
        
        this.visualizationInterval = setInterval(() => {
            bars.forEach(bar => {
                const height = Math.random() * 100;
                bar.style.height = `${height}%`;
            });
        }, 100);
    }
    
    stopAudioVisualization() {
        if (this.visualizationInterval) {
            clearInterval(this.visualizationInterval);
            this.visualizationInterval = null;
        }
        
        const bars = this.elements.audioVisualizer.querySelectorAll('.bar');
        bars.forEach(bar => {
            bar.style.height = '20%';
        });
    }
    
    addMessage(sender, text, metadata = {}) {
        const messageContainer = this.elements.conversationMessages;
        const messageElement = document.createElement('div');
        messageElement.className = `message ${sender}`;
        
        const timestamp = new Date().toLocaleTimeString();
        const showTimestamps = this.elements.timestampsToggle.checked;
        
        messageElement.innerHTML = `
            <div class="message-avatar">
                <i class="fas ${sender === 'user' ? 'fa-user' : 'fa-robot'}"></i>
            </div>
            <div class="message-content">
                <div class="message-bubble">
                    <p>${text}</p>
                    ${metadata.confidence ? `<div class="confidence">Confidence: ${(metadata.confidence * 100).toFixed(1)}%</div>` : ''}
                </div>
                ${showTimestamps ? `<div class="message-meta"><span class="timestamp">${timestamp}</span></div>` : ''}
            </div>
        `;
        
        // Remove welcome message if it exists
        const welcomeMessage = messageContainer.querySelector('.welcome-message');
        if (welcomeMessage) {
            welcomeMessage.remove();
        }
        
        messageContainer.appendChild(messageElement);
        messageContainer.scrollTop = messageContainer.scrollHeight;
        
        // Store in conversation history
        this.conversation.push({
            sender,
            text,
            timestamp: new Date().toISOString(),
            metadata
        });
    }
    
    showTypingIndicator() {
        this.elements.typingIndicator.style.display = 'flex';
        this.elements.conversationMessages.scrollTop = this.elements.conversationMessages.scrollHeight;
    }
    
    hideTypingIndicator() {
        this.elements.typingIndicator.style.display = 'none';
    }
    
    async createNewSession() {
        if (this.isRecording) {
            await this.stopRecording();
        }
        
        this.sessionId = null;
        this.conversation = [];
        this.clearConversation();
        
        if (this.isConnected) {
            this.disconnect();
        }
        
        this.connect();
    }
    
    clearConversation() {
        this.conversation = [];
        this.elements.conversationMessages.innerHTML = `
            <div class="welcome-message">
                <div class="message assistant">
                    <div class="message-avatar">
                        <i class="fas fa-robot"></i>
                    </div>
                    <div class="message-content">
                        <div class="message-bubble">
                            <p>Conversation cleared. Ready for a new chat!</p>
                        </div>
                        <div class="message-meta">
                            <span class="timestamp">Now</span>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }
    
    async loadVoices() {
        try {
            const voices = await this.voiceManager.getVoices();
            const voiceSelect = this.elements.voiceSelect;
            
            // Clear existing options except default
            voiceSelect.innerHTML = '<option value="default">Default Voice</option>';
            
            // Add custom voices
            voices.forEach(voice => {
                const option = document.createElement('option');
                option.value = voice.id;
                option.textContent = voice.name;
                voiceSelect.appendChild(option);
            });
            
        } catch (error) {
            console.error('Failed to load voices:', error);
        }
    }
    
    setVoice(voiceId) {
        if (this.isConnected && this.websocket.readyState === WebSocket.OPEN) {
            this.websocket.send(JSON.stringify({
                type: 'set_voice',
                voice_id: voiceId
            }));
        }
    }
    
    setLanguage(language) {
        if (this.isConnected && this.websocket.readyState === WebSocket.OPEN) {
            this.websocket.send(JSON.stringify({
                type: 'set_language',
                language: language
            }));
        }
    }
    
    exportConversation() {
        const data = {
            sessionId: this.sessionId,
            timestamp: new Date().toISOString(),
            conversation: this.conversation
        };
        
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = `lexai-conversation-${new Date().toISOString().split('T')[0]}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        
        URL.revokeObjectURL(url);
        this.showNotification('Conversation exported', 'success');
    }
    
    updateConnectionStatus(status, text) {
        this.elements.statusIndicator.className = `status-indicator status-${status}`;
        this.elements.statusText.textContent = text;
    }
    
    showLoading(text) {
        this.elements.loadingText.textContent = text;
        this.elements.loadingOverlay.style.display = 'flex';
    }
    
    hideLoading() {
        this.elements.loadingOverlay.style.display = 'none';
    }
    
    showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.innerHTML = `
            <div class="notification-content">
                <i class="fas ${this.getNotificationIcon(type)}"></i>
                <span>${message}</span>
            </div>
            <button class="notification-close">
                <i class="fas fa-times"></i>
            </button>
        `;
        
        this.elements.notificationContainer.appendChild(notification);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            notification.remove();
        }, 5000);
        
        // Add close button handler
        notification.querySelector('.notification-close').addEventListener('click', () => {
            notification.remove();
        });
    }
    
    getNotificationIcon(type) {
        const icons = {
            success: 'fa-check-circle',
            error: 'fa-exclamation-circle',
            warning: 'fa-exclamation-triangle',
            info: 'fa-info-circle'
        };
        return icons[type] || icons.info;
    }
    
    toggleSettings() {
        const settingsContent = this.elements.settingsContent;
        const isOpen = settingsContent.style.display === 'block';
        settingsContent.style.display = isOpen ? 'none' : 'block';
    }
    
    openVoiceModal() {
        this.elements.voiceModal.style.display = 'flex';
        this.voiceManager.refreshVoiceList();
    }
    
    closeVoiceModal() {
        this.elements.voiceModal.style.display = 'none';
    }
    
    handleKeyboardShortcuts(e) {
        // Space bar to toggle recording
        if (e.code === 'Space' && !e.target.matches('input, textarea')) {
            e.preventDefault();
            this.toggleRecording();
        }
        
        // Escape to stop recording
        if (e.code === 'Escape' && this.isRecording) {
            this.stopRecording();
        }
        
        // Ctrl+N for new session
        if (e.ctrlKey && e.code === 'KeyN') {
            e.preventDefault();
            this.createNewSession();
        }
        
        // Ctrl+E to export conversation
        if (e.ctrlKey && e.code === 'KeyE') {
            e.preventDefault();
            this.exportConversation();
        }
    }
    
    handleOnline() {
        this.showNotification('Connection restored', 'success');
        if (!this.isConnected) {
            this.connect();
        }
    }
    
    handleOffline() {
        this.showNotification('Connection lost', 'warning');
    }
    
    handleConnectionError(error) {
        console.error('Connection error:', error);
        this.hideLoading();
        this.updateConnectionStatus('error', 'Connection error');
        this.showNotification('Failed to connect to server', 'error');
    }
    
    disconnect() {
        if (this.websocket) {
            this.websocket.close();
            this.websocket = null;
        }
        this.isConnected = false;
        this.stopHeartbeat();
        
        if (this.isRecording) {
            this.stopRecording();
        }
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.lexaiApp = new LexAIApp();
});