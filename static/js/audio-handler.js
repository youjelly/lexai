/**
 * Audio Handler for LexAI Web Client
 * Handles microphone capture, audio processing, and WebSocket streaming
 */

class AudioHandler {
    constructor() {
        this.mediaRecorder = null;
        this.audioStream = null;
        this.audioContext = null;
        this.analyser = null;
        this.microphone = null;
        this.isRecording = false;
        this.audioChunks = [];
        
        // Audio configuration
        this.config = {
            sampleRate: 16000,        // 16kHz for speech recognition
            channels: 1,              // Mono audio
            bitDepth: 16,             // 16-bit audio
            chunkSize: 4096,          // Audio chunk size
            bufferSize: 2048,         // Buffer size for processing
            format: 'webm;codecs=opus' // Audio format
        };
        
        // Callback for streaming audio data
        this.onAudioData = null;
        
        // Audio level monitoring
        this.audioLevels = new Float32Array(128);
        this.levelCallback = null;
    }
    
    async init() {
        try {
            // Check for WebRTC support
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                throw new Error('WebRTC not supported in this browser');
            }
            
            // Check for MediaRecorder support
            if (!window.MediaRecorder) {
                throw new Error('MediaRecorder not supported in this browser');
            }
            
            // Check for AudioContext support
            const AudioContext = window.AudioContext || window.webkitAudioContext;
            if (!AudioContext) {
                throw new Error('Web Audio API not supported in this browser');
            }
            
            console.log('Audio handler initialized successfully');
            return true;
            
        } catch (error) {
            console.error('Audio handler initialization failed:', error);
            throw error;
        }
    }
    
    async requestMicrophonePermission() {
        try {
            const constraints = {
                audio: {
                    sampleRate: this.config.sampleRate,
                    channelCount: this.config.channels,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                },
                video: false
            };
            
            this.audioStream = await navigator.mediaDevices.getUserMedia(constraints);
            console.log('Microphone permission granted');
            return this.audioStream;
            
        } catch (error) {
            console.error('Microphone permission denied:', error);
            throw new Error('Microphone access denied. Please allow microphone access and try again.');
        }
    }
    
    async setupAudioContext() {
        try {
            const AudioContext = window.AudioContext || window.webkitAudioContext;
            this.audioContext = new AudioContext({
                sampleRate: this.config.sampleRate
            });
            
            // Resume audio context if suspended (required by some browsers)
            if (this.audioContext.state === 'suspended') {
                await this.audioContext.resume();
            }
            
            // Create analyser for audio visualization
            this.analyser = this.audioContext.createAnalyser();
            this.analyser.fftSize = 256;
            this.analyser.smoothingTimeConstant = 0.8;
            
            // Connect microphone to analyser
            this.microphone = this.audioContext.createMediaStreamSource(this.audioStream);
            this.microphone.connect(this.analyser);
            
            console.log('Audio context setup complete');
            
        } catch (error) {
            console.error('Audio context setup failed:', error);
            throw error;
        }
    }
    
    async startRecording(onAudioData) {
        if (this.isRecording) {
            console.warn('Recording already in progress');
            return;
        }
        
        try {
            // Store callback
            this.onAudioData = onAudioData;
            
            // Request microphone access
            await this.requestMicrophonePermission();
            
            // Setup audio context
            await this.setupAudioContext();
            
            // Setup MediaRecorder
            await this.setupMediaRecorder();
            
            // Start recording
            this.mediaRecorder.start(100); // Send data every 100ms
            this.isRecording = true;
            
            // Start audio level monitoring
            this.startAudioLevelMonitoring();
            
            console.log('Recording started');
            
        } catch (error) {
            console.error('Failed to start recording:', error);
            await this.cleanup();
            throw error;
        }
    }
    
    async setupMediaRecorder() {
        try {
            // Determine the best supported format
            const mimeType = this.getSupportedMimeType();
            
            this.mediaRecorder = new MediaRecorder(this.audioStream, {
                mimeType: mimeType,
                audioBitsPerSecond: 16000 // 16kbps for speech
            });
            
            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data && event.data.size > 0) {
                    this.processAudioChunk(event.data);
                }
            };
            
            this.mediaRecorder.onerror = (event) => {
                console.error('MediaRecorder error:', event.error);
                this.stopRecording();
            };
            
            console.log('MediaRecorder setup complete with format:', mimeType);
            
        } catch (error) {
            console.error('MediaRecorder setup failed:', error);
            throw error;
        }
    }
    
    getSupportedMimeType() {
        const types = [
            'audio/webm;codecs=opus',
            'audio/webm',
            'audio/ogg;codecs=opus',
            'audio/ogg',
            'audio/wav'
        ];
        
        for (const type of types) {
            if (MediaRecorder.isTypeSupported(type)) {
                return type;
            }
        }
        
        throw new Error('No supported audio MIME type found');
    }
    
    async processAudioChunk(audioBlob) {
        try {
            // Convert blob to ArrayBuffer for streaming
            const arrayBuffer = await audioBlob.arrayBuffer();
            
            // Send to callback if provided
            if (this.onAudioData && typeof this.onAudioData === 'function') {
                this.onAudioData(arrayBuffer);
            }
            
            // Store for potential later use
            this.audioChunks.push(audioBlob);
            
        } catch (error) {
            console.error('Failed to process audio chunk:', error);
        }
    }
    
    startAudioLevelMonitoring() {
        if (!this.analyser) return;
        
        const updateLevels = () => {
            if (!this.isRecording) return;
            
            this.analyser.getFloatFrequencyData(this.audioLevels);
            
            // Calculate average level
            let sum = 0;
            for (let i = 0; i < this.audioLevels.length; i++) {
                sum += this.audioLevels[i];
            }
            const average = sum / this.audioLevels.length;
            
            // Convert to 0-100 scale
            const level = Math.max(0, Math.min(100, (average + 100) * 2));
            
            // Call level callback if provided
            if (this.levelCallback) {
                this.levelCallback(level, this.audioLevels);
            }
            
            // Continue monitoring
            requestAnimationFrame(updateLevels);
        };
        
        updateLevels();
    }
    
    async stopRecording() {
        if (!this.isRecording) {
            console.warn('Not currently recording');
            return;
        }
        
        try {
            this.isRecording = false;
            
            // Stop MediaRecorder
            if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
                this.mediaRecorder.stop();
            }
            
            // Cleanup
            await this.cleanup();
            
            console.log('Recording stopped');
            
        } catch (error) {
            console.error('Failed to stop recording:', error);
            throw error;
        }
    }
    
    async cleanup() {
        try {
            // Stop audio stream
            if (this.audioStream) {
                this.audioStream.getTracks().forEach(track => track.stop());
                this.audioStream = null;
            }
            
            // Close audio context
            if (this.audioContext && this.audioContext.state !== 'closed') {
                await this.audioContext.close();
                this.audioContext = null;
            }
            
            // Reset references
            this.mediaRecorder = null;
            this.analyser = null;
            this.microphone = null;
            this.audioChunks = [];
            this.onAudioData = null;
            
        } catch (error) {
            console.error('Cleanup failed:', error);
        }
    }
    
    setAudioLevelCallback(callback) {
        this.levelCallback = callback;
    }
    
    getRecordedAudio() {
        if (this.audioChunks.length === 0) {
            return null;
        }
        
        // Combine all audio chunks into a single blob
        const mimeType = this.getSupportedMimeType();
        return new Blob(this.audioChunks, { type: mimeType });
    }
    
    clearRecordedAudio() {
        this.audioChunks = [];
    }
    
    // Convert audio to different formats if needed
    async convertAudioFormat(audioBlob, targetFormat = 'wav') {
        try {
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const arrayBuffer = await audioBlob.arrayBuffer();
            const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
            
            if (targetFormat === 'wav') {
                return this.audioBufferToWav(audioBuffer);
            }
            
            throw new Error(`Unsupported target format: ${targetFormat}`);
            
        } catch (error) {
            console.error('Audio format conversion failed:', error);
            throw error;
        }
    }
    
    audioBufferToWav(audioBuffer) {
        const numberOfChannels = audioBuffer.numberOfChannels;
        const sampleRate = audioBuffer.sampleRate;
        const format = 1; // PCM
        const bitDepth = 16;
        
        const bytesPerSample = bitDepth / 8;
        const blockAlign = numberOfChannels * bytesPerSample;
        
        const buffer = audioBuffer.getChannelData(0);
        const arrayBuffer = new ArrayBuffer(44 + buffer.length * bytesPerSample);
        const view = new DataView(arrayBuffer);
        
        // WAV header
        const writeString = (offset, string) => {
            for (let i = 0; i < string.length; i++) {
                view.setUint8(offset + i, string.charCodeAt(i));
            }
        };
        
        writeString(0, 'RIFF');
        view.setUint32(4, 36 + buffer.length * bytesPerSample, true);
        writeString(8, 'WAVE');
        writeString(12, 'fmt ');
        view.setUint32(16, 16, true);
        view.setUint16(20, format, true);
        view.setUint16(22, numberOfChannels, true);
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, sampleRate * blockAlign, true);
        view.setUint16(32, blockAlign, true);
        view.setUint16(34, bitDepth, true);
        writeString(36, 'data');
        view.setUint32(40, buffer.length * bytesPerSample, true);
        
        // Convert audio data
        let offset = 44;
        for (let i = 0; i < buffer.length; i++) {
            const sample = Math.max(-1, Math.min(1, buffer[i]));
            view.setInt16(offset, sample * 0x7FFF, true);
            offset += 2;
        }
        
        return new Blob([arrayBuffer], { type: 'audio/wav' });
    }
    
    // Test audio functionality
    async testAudio() {
        try {
            console.log('Testing audio functionality...');
            
            // Test microphone access
            await this.requestMicrophonePermission();
            console.log('✓ Microphone access granted');
            
            // Test audio context
            await this.setupAudioContext();
            console.log('✓ Audio context created');
            
            // Test MediaRecorder
            await this.setupMediaRecorder();
            console.log('✓ MediaRecorder setup successful');
            
            // Cleanup
            await this.cleanup();
            console.log('✓ Audio test completed successfully');
            
            return true;
            
        } catch (error) {
            console.error('Audio test failed:', error);
            await this.cleanup();
            throw error;
        }
    }
    
    // Get audio input devices
    async getAudioInputDevices() {
        try {
            const devices = await navigator.mediaDevices.enumerateDevices();
            return devices.filter(device => device.kind === 'audioinput');
        } catch (error) {
            console.error('Failed to get audio input devices:', error);
            return [];
        }
    }
    
    // Switch audio input device
    async switchAudioDevice(deviceId) {
        try {
            if (this.isRecording) {
                throw new Error('Cannot switch audio device while recording');
            }
            
            const constraints = {
                audio: {
                    deviceId: deviceId ? { exact: deviceId } : undefined,
                    sampleRate: this.config.sampleRate,
                    channelCount: this.config.channels,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                },
                video: false
            };
            
            // Stop current stream if exists
            if (this.audioStream) {
                this.audioStream.getTracks().forEach(track => track.stop());
            }
            
            // Get new stream
            this.audioStream = await navigator.mediaDevices.getUserMedia(constraints);
            console.log('Switched to audio device:', deviceId);
            
        } catch (error) {
            console.error('Failed to switch audio device:', error);
            throw error;
        }
    }
}