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
        
        // Voice Activity Detection
        this.vadEnabled = true;
        this.voiceActivityCallback = null;
        this.isVoiceActive = false;
        this.vadThreshold = 0.01;  // Energy threshold for voice detection
        this.vadSmoothingFrames = 3;  // Number of frames to smooth over
        this.vadHistory = [];  // History of voice activity
        this.silenceFrames = 0;
        this.speechFrames = 0;
        this.minSpeechFrames = 3;  // Minimum frames to consider as speech
        this.minSilenceFrames = 10;  // Minimum frames to consider as silence
        this.energySmoothing = 0.95;  // Exponential smoothing factor
        this.smoothedEnergy = 0;
    }
    
    async init() {
        try {
            // Check for WebRTC support
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                console.warn('navigator.mediaDevices not available. This might be due to:');
                console.warn('1. Using HTTP instead of HTTPS');
                console.warn('2. Browser security settings');
                console.warn('navigator.mediaDevices:', navigator.mediaDevices);
                // Don't throw error here, let it fail when actually requesting permission
                // throw new Error('WebRTC not supported in this browser');
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
                    // Don't specify sampleRate in constraints - let browser use default
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
            // Don't specify sample rate - let it match the microphone stream
            this.audioContext = new AudioContext();
            
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
            
            // Setup raw audio processing instead of MediaRecorder
            await this.setupRawAudioProcessing();
            
            // Start recording
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
            
            // Disconnect script processor
            if (this.scriptProcessor) {
                this.scriptProcessor.disconnect();
                this.scriptProcessor = null;
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
    
    async setupRawAudioProcessing() {
        try {
            // Create a script processor node (deprecated but still works)
            // Using 4096 sample buffer size for ~256ms chunks at 16kHz
            const bufferSize = 4096;
            this.scriptProcessor = this.audioContext.createScriptProcessor(bufferSize, 1, 1);
            
            // Connect microphone -> script processor (no destination to avoid echo)
            this.microphone.connect(this.scriptProcessor);
            // Note: Not connecting to destination to avoid echo
            
            // Calculate downsampling ratio (browser usually 44.1kHz or 48kHz -> 16kHz)
            const targetSampleRate = 16000;
            const sourceSampleRate = this.audioContext.sampleRate;
            const downsampleRatio = sourceSampleRate / targetSampleRate;
            
            console.log(`Audio context sample rate: ${sourceSampleRate}Hz, downsampling to ${targetSampleRate}Hz`);
            
            // Process audio chunks
            this.scriptProcessor.onaudioprocess = (event) => {
                if (!this.isRecording) return;
                
                // Get raw float32 samples
                const inputData = event.inputBuffer.getChannelData(0);
                
                // Perform VAD if enabled
                if (this.vadEnabled) {
                    this.processVAD(inputData);
                }
                
                // Downsample to 16kHz
                const downsampledLength = Math.floor(inputData.length / downsampleRatio);
                const downsampled = new Float32Array(downsampledLength);
                
                for (let i = 0; i < downsampledLength; i++) {
                    const sourceIndex = Math.floor(i * downsampleRatio);
                    downsampled[i] = inputData[sourceIndex];
                }
                
                // Convert float32 to int16 PCM
                const pcm16 = new Int16Array(downsampledLength);
                for (let i = 0; i < downsampledLength; i++) {
                    // Clamp to [-1, 1] range
                    const sample = Math.max(-1, Math.min(1, downsampled[i]));
                    // Convert to int16
                    pcm16[i] = Math.floor(sample * 0x7FFF);
                }
                
                // Send PCM data as ArrayBuffer
                if (this.onAudioData) {
                    this.onAudioData(pcm16.buffer);
                }
            };
            
            console.log('Raw audio processing setup complete');
            
        } catch (error) {
            console.error('Raw audio processing setup failed:', error);
            throw error;
        }
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
    
    // Process Voice Activity Detection
    processVAD(inputData) {
        // Continue processing VAD even during recording to detect when to stop
        
        // Calculate energy of the audio frame
        let energy = 0;
        for (let i = 0; i < inputData.length; i++) {
            energy += inputData[i] * inputData[i];
        }
        energy = energy / inputData.length;
        
        // Apply exponential smoothing
        this.smoothedEnergy = this.energySmoothing * this.smoothedEnergy + 
                             (1 - this.energySmoothing) * energy;
        
        // Determine if this frame contains voice
        const frameHasVoice = this.smoothedEnergy > this.vadThreshold;
        
        // Update history
        this.vadHistory.push(frameHasVoice);
        if (this.vadHistory.length > this.vadSmoothingFrames) {
            this.vadHistory.shift();
        }
        
        // Count recent voice/silence frames
        const recentVoiceFrames = this.vadHistory.filter(v => v).length;
        const voiceDetected = recentVoiceFrames >= Math.floor(this.vadSmoothingFrames * 0.6);
        
        // Update speech/silence frame counters
        if (voiceDetected) {
            this.speechFrames++;
            this.silenceFrames = 0;
        } else {
            this.silenceFrames++;
            this.speechFrames = 0;
        }
        
        // Determine overall voice activity state with hysteresis
        const wasActive = this.isVoiceActive;
        
        if (!wasActive && this.speechFrames >= this.minSpeechFrames) {
            // Transition to active
            this.isVoiceActive = true;
            console.log('Voice activity started');
            if (this.voiceActivityCallback) {
                this.voiceActivityCallback(true);
            }
        } else if (wasActive && this.silenceFrames >= this.minSilenceFrames) {
            // Transition to inactive
            this.isVoiceActive = false;
            console.log('Voice activity ended');
            if (this.voiceActivityCallback) {
                this.voiceActivityCallback(false);
            }
        }
    }
    
    // Set VAD callback
    setVoiceActivityCallback(callback) {
        this.voiceActivityCallback = callback;
    }
    
    // Enable/disable VAD
    setVADEnabled(enabled) {
        this.vadEnabled = enabled;
        console.log(`VAD ${enabled ? 'enabled' : 'disabled'}`);
    }
    
    // Get current voice activity state
    getVoiceActivityState() {
        return this.isVoiceActive;
    }
    
    // Start VAD monitoring without recording
    async startVADMonitoring() {
        if (this.isRecording) {
            console.warn('Cannot start VAD monitoring while recording');
            return;
        }
        
        try {
            // Request microphone access
            await this.requestMicrophonePermission();
            
            // Setup audio context
            await this.setupAudioContext();
            
            // Setup script processor for VAD only
            await this.setupVADProcessing();
            
            console.log('VAD monitoring started');
            
        } catch (error) {
            console.error('Failed to start VAD monitoring:', error);
            throw error;
        }
    }
    
    async setupVADProcessing() {
        try {
            // Create a script processor node for VAD
            const bufferSize = 2048;  // Smaller buffer for faster VAD response
            this.vadProcessor = this.audioContext.createScriptProcessor(bufferSize, 1, 1);
            
            // Connect microphone -> script processor (no destination to avoid echo)
            this.microphone.connect(this.vadProcessor);
            
            // Process audio for VAD only
            this.vadProcessor.onaudioprocess = (event) => {
                // Get raw float32 samples
                const inputData = event.inputBuffer.getChannelData(0);
                
                // Perform VAD
                if (this.vadEnabled) {
                    this.processVAD(inputData);
                }
            };
            
            console.log('VAD processing setup complete');
            
        } catch (error) {
            console.error('VAD processing setup failed:', error);
            throw error;
        }
    }
}