"""
LexAI WebSocket module
Real-time audio streaming with WebSocket support
"""

from .audio_streamer import (
    AudioStreamer,
    MessageType,
    StreamingConfig,
    create_audio_streamer
)

from .session_manager import (
    SessionManager,
    SessionInfo,
    session_manager
)

from .audio_handler import (
    AudioHandler,
    AudioFormat,
    AudioBuffer
)

from .connection_manager import (
    ConnectionManager,
    ConnectionInfo,
    connection_manager,
    websocket_handler,
    start_websocket_server
)

__all__ = [
    # Audio streaming
    'AudioStreamer',
    'MessageType',
    'StreamingConfig',
    'create_audio_streamer',
    
    # Session management
    'SessionManager',
    'SessionInfo',
    'session_manager',
    
    # Audio handling
    'AudioHandler',
    'AudioFormat',
    'AudioBuffer',
    
    # Connection management
    'ConnectionManager',
    'ConnectionInfo',
    'connection_manager',
    'websocket_handler',
    'start_websocket_server'
]