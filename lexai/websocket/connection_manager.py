"""
WebSocket connection management for LexAI
Handles connection lifecycle, authentication, and routing
"""

import asyncio
import json
import time
import jwt
from typing import Dict, Set, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import websockets
from websockets.server import WebSocketServerProtocol
from websockets.exceptions import ConnectionClosed, InvalidStatusCode

from .audio_streamer import AudioStreamer, create_audio_streamer
from .session_manager import session_manager
from .audio_handler import AudioHandler
from ..utils.logging import get_logger
from config import settings

logger = get_logger(__name__)


@dataclass
class ConnectionInfo:
    """Information about a WebSocket connection"""
    connection_id: str
    websocket: WebSocketServerProtocol
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    connected_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    
    # Connection metadata
    client_ip: str = ""
    user_agent: str = ""
    origin: str = ""
    
    # Authentication
    is_authenticated: bool = False
    auth_token: Optional[str] = None
    
    # Handlers
    audio_streamer: Optional[AudioStreamer] = None
    audio_handler: Optional[AudioHandler] = None
    
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = time.time()


class ConnectionManager:
    """Manages WebSocket connections"""
    
    def __init__(self):
        # Active connections
        self.connections: Dict[str, ConnectionInfo] = {}
        self.user_connections: Dict[str, Set[str]] = {}  # user_id -> connection_ids
        
        # Connection limits
        self.max_connections = 1000
        self.max_connections_per_user = 5
        self.connection_timeout = 300  # 5 minutes
        
        # Authentication
        self.auth_required = False  # Set to True in production
        self.jwt_secret = settings.HF_TOKEN  # Use proper secret in production
        
        # Handlers
        self.message_handlers: Dict[str, Callable] = {}
        
        # Monitoring
        self.stats = {
            "total_connections": 0,
            "failed_connections": 0,
            "authenticated_connections": 0,
            "messages_received": 0,
            "messages_sent": 0,
            "bytes_received": 0,
            "bytes_sent": 0
        }
        
        # Cleanup task
        self.cleanup_task = None
    
    async def initialize(self):
        """Initialize connection manager"""
        # Initialize session manager
        await session_manager.initialize()
        
        # Start cleanup task
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info("Connection manager initialized")
    
    async def handle_connection(
        self,
        websocket: WebSocketServerProtocol,
        path: str
    ):
        """Handle new WebSocket connection"""
        connection_id = f"conn_{int(time.time() * 1000)}_{len(self.connections)}"
        
        # Check connection limit
        if len(self.connections) >= self.max_connections:
            await websocket.close(1008, "Server at capacity")
            self.stats["failed_connections"] += 1
            return
        
        # Extract connection metadata
        headers = websocket.request_headers
        client_ip = websocket.remote_address[0] if websocket.remote_address else "unknown"
        
        # Create connection info
        connection = ConnectionInfo(
            connection_id=connection_id,
            websocket=websocket,
            client_ip=client_ip,
            user_agent=headers.get("User-Agent", ""),
            origin=headers.get("Origin", "")
        )
        
        # Store connection
        self.connections[connection_id] = connection
        self.stats["total_connections"] += 1
        
        logger.info(f"New connection {connection_id} from {client_ip}")
        
        try:
            # Handle authentication if required
            if self.auth_required:
                authenticated = await self._authenticate_connection(connection)
                if not authenticated:
                    await websocket.close(1008, "Authentication failed")
                    return
            
            # Send welcome message
            await self._send_welcome(connection)
            
            # Handle messages
            await self._handle_messages(connection)
            
        except ConnectionClosed:
            logger.info(f"Connection {connection_id} closed")
        except Exception as e:
            logger.error(f"Connection {connection_id} error: {e}")
            await websocket.close(1011, "Internal error")
        finally:
            await self._cleanup_connection(connection)
    
    async def _authenticate_connection(self, connection: ConnectionInfo) -> bool:
        """Authenticate WebSocket connection"""
        try:
            # Wait for auth message
            websocket = connection.websocket
            auth_timeout = 10.0  # 10 seconds to authenticate
            
            auth_message = await asyncio.wait_for(
                websocket.recv(),
                timeout=auth_timeout
            )
            
            data = json.loads(auth_message)
            
            if data.get("type") != "auth":
                await self._send_error(connection, "First message must be auth")
                return False
            
            token = data.get("token")
            if not token:
                await self._send_error(connection, "Auth token required")
                return False
            
            # Verify token
            try:
                payload = jwt.decode(
                    token,
                    self.jwt_secret,
                    algorithms=["HS256"]
                )
                
                connection.user_id = payload.get("user_id")
                connection.auth_token = token
                connection.is_authenticated = True
                
                # Track user connections
                if connection.user_id:
                    if connection.user_id not in self.user_connections:
                        self.user_connections[connection.user_id] = set()
                    
                    # Check per-user limit
                    if len(self.user_connections[connection.user_id]) >= self.max_connections_per_user:
                        await self._send_error(connection, "User connection limit exceeded")
                        return False
                    
                    self.user_connections[connection.user_id].add(connection.connection_id)
                
                self.stats["authenticated_connections"] += 1
                
                # Send auth success
                await self._send_message(connection, {
                    "type": "auth_success",
                    "user_id": connection.user_id
                })
                
                logger.info(f"Connection {connection.connection_id} authenticated as {connection.user_id}")
                return True
                
            except jwt.InvalidTokenError as e:
                await self._send_error(connection, f"Invalid token: {e}")
                return False
                
        except asyncio.TimeoutError:
            await self._send_error(connection, "Authentication timeout")
            return False
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            await self._send_error(connection, "Authentication failed")
            return False
    
    async def _handle_messages(self, connection: ConnectionInfo):
        """Handle messages from connection"""
        websocket = connection.websocket
        
        async for message in websocket:
            try:
                connection.update_activity()
                self.stats["messages_received"] += 1
                
                # Handle binary audio data
                if isinstance(message, bytes):
                    self.stats["bytes_received"] += len(message)
                    
                    if connection.audio_streamer:
                        await connection.audio_streamer._handle_audio_data(message)
                    else:
                        await self._send_error(connection, "No active audio session")
                    continue
                
                # Handle JSON messages
                self.stats["bytes_received"] += len(message.encode())
                data = json.loads(message)
                message_type = data.get("type")
                
                if message_type == "start_session":
                    await self._handle_start_session(connection, data)
                elif message_type == "end_session":
                    await self._handle_end_session(connection)
                elif message_type == "ping":
                    await self._handle_ping(connection)
                else:
                    # Forward to audio streamer if active
                    if connection.audio_streamer:
                        # Recreate message for streamer
                        await websocket.send(message)
                    else:
                        await self._send_error(connection, f"Unknown message type: {message_type}")
                
            except json.JSONDecodeError:
                await self._send_error(connection, "Invalid JSON")
            except Exception as e:
                logger.error(f"Message handling error: {e}")
                await self._send_error(connection, str(e))
    
    async def _handle_start_session(self, connection: ConnectionInfo, data: Dict[str, Any]):
        """Handle session start request"""
        if connection.session_id:
            await self._send_error(connection, "Session already active")
            return
        
        try:
            # Create session
            session = await session_manager.create_session(
                user_id=connection.user_id,
                client_info={
                    "ip": connection.client_ip,
                    "user_agent": connection.user_agent,
                    "origin": connection.origin
                }
            )
            
            connection.session_id = session.session_id
            
            # Create audio streamer
            connection.audio_streamer = await create_audio_streamer(session.session_id)
            connection.audio_handler = AudioHandler(session.session_id)
            
            # Register with session manager
            session_manager.register_handler(session.session_id, connection.audio_streamer)
            
            # Configure from request
            config = data.get("config", {})
            if config:
                await connection.audio_streamer._handle_init({"config": config}, connection.websocket)
            
            # Send session started
            await self._send_message(connection, {
                "type": "session_started",
                "session_id": session.session_id,
                "capabilities": {
                    "audio_formats": ["pcm16", "pcm32", "float32"],
                    "sample_rates": [16000, 22050, 44100, 48000],
                    "streaming": True,
                    "voice_cloning": True
                }
            })
            
            # Start streaming handler
            asyncio.create_task(
                connection.audio_streamer.handle_connection(
                    connection.websocket,
                    ""
                )
            )
            
            logger.info(f"Started session {session.session_id} for connection {connection.connection_id}")
            
        except Exception as e:
            logger.error(f"Failed to start session: {e}")
            await self._send_error(connection, f"Failed to start session: {e}")
    
    async def _handle_end_session(self, connection: ConnectionInfo):
        """Handle session end request"""
        if not connection.session_id:
            await self._send_error(connection, "No active session")
            return
        
        try:
            # End session
            await session_manager.end_session(connection.session_id)
            
            # Clean up handlers
            if connection.audio_handler:
                await connection.audio_handler.cleanup()
            
            connection.session_id = None
            connection.audio_streamer = None
            connection.audio_handler = None
            
            # Send confirmation
            await self._send_message(connection, {
                "type": "session_ended"
            })
            
            logger.info(f"Ended session for connection {connection.connection_id}")
            
        except Exception as e:
            logger.error(f"Failed to end session: {e}")
            await self._send_error(connection, f"Failed to end session: {e}")
    
    async def _handle_ping(self, connection: ConnectionInfo):
        """Handle ping message"""
        await self._send_message(connection, {
            "type": "pong",
            "timestamp": time.time()
        })
    
    async def _send_welcome(self, connection: ConnectionInfo):
        """Send welcome message to new connection"""
        await self._send_message(connection, {
            "type": "welcome",
            "connection_id": connection.connection_id,
            "server_time": time.time(),
            "auth_required": self.auth_required
        })
    
    async def _send_message(self, connection: ConnectionInfo, data: Dict[str, Any]):
        """Send message to connection"""
        try:
            message = json.dumps(data)
            await connection.websocket.send(message)
            self.stats["messages_sent"] += 1
            self.stats["bytes_sent"] += len(message.encode())
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
    
    async def _send_error(self, connection: ConnectionInfo, error: str):
        """Send error message to connection"""
        await self._send_message(connection, {
            "type": "error",
            "error": error,
            "timestamp": time.time()
        })
    
    async def _cleanup_connection(self, connection: ConnectionInfo):
        """Clean up connection resources"""
        connection_id = connection.connection_id
        
        # End any active session
        if connection.session_id:
            try:
                await session_manager.end_session(connection.session_id)
            except Exception as e:
                logger.error(f"Failed to end session on cleanup: {e}")
        
        # Clean up handlers
        if connection.audio_handler:
            await connection.audio_handler.cleanup()
        
        # Remove from user connections
        if connection.user_id and connection.user_id in self.user_connections:
            self.user_connections[connection.user_id].discard(connection_id)
            if not self.user_connections[connection.user_id]:
                del self.user_connections[connection.user_id]
        
        # Remove connection
        if connection_id in self.connections:
            del self.connections[connection_id]
        
        logger.info(f"Cleaned up connection {connection_id}")
    
    async def _cleanup_loop(self):
        """Periodic cleanup of inactive connections"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                now = time.time()
                inactive_connections = []
                
                for conn_id, conn in self.connections.items():
                    # Check for timeout
                    if now - conn.last_activity > self.connection_timeout:
                        inactive_connections.append(conn_id)
                
                # Close inactive connections
                for conn_id in inactive_connections:
                    conn = self.connections.get(conn_id)
                    if conn:
                        logger.info(f"Closing inactive connection {conn_id}")
                        await conn.websocket.close(1000, "Idle timeout")
                
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
    
    async def broadcast_to_user(self, user_id: str, message: Dict[str, Any]):
        """Broadcast message to all connections for a user"""
        if user_id not in self.user_connections:
            return
        
        for conn_id in self.user_connections[user_id]:
            conn = self.connections.get(conn_id)
            if conn:
                await self._send_message(conn, message)
    
    async def close_user_connections(self, user_id: str, reason: str = "User logout"):
        """Close all connections for a user"""
        if user_id not in self.user_connections:
            return
        
        conn_ids = list(self.user_connections[user_id])
        for conn_id in conn_ids:
            conn = self.connections.get(conn_id)
            if conn:
                await conn.websocket.close(1000, reason)
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        return {
            "active_connections": len(self.connections),
            "authenticated_users": len(self.user_connections),
            "total_stats": self.stats,
            "connections_by_user": {
                user_id: len(conns)
                for user_id, conns in self.user_connections.items()
            },
            "recent_connections": [
                {
                    "connection_id": conn.connection_id,
                    "user_id": conn.user_id,
                    "connected_at": datetime.fromtimestamp(conn.connected_at).isoformat(),
                    "last_activity": datetime.fromtimestamp(conn.last_activity).isoformat(),
                    "session_id": conn.session_id,
                    "client_ip": conn.client_ip
                }
                for conn in sorted(
                    self.connections.values(),
                    key=lambda c: c.connected_at,
                    reverse=True
                )[:10]
            ]
        }
    
    async def cleanup(self):
        """Clean up connection manager"""
        # Cancel cleanup task
        if self.cleanup_task:
            self.cleanup_task.cancel()
        
        # Close all connections
        for conn in list(self.connections.values()):
            await conn.websocket.close(1001, "Server shutdown")
        
        # Clean up session manager
        await session_manager.cleanup()
        
        logger.info("Connection manager cleanup complete")


# Global connection manager instance
connection_manager = ConnectionManager()


async def websocket_handler(websocket: WebSocketServerProtocol, path: str):
    """Main WebSocket handler for server"""
    await connection_manager.handle_connection(websocket, path)


async def start_websocket_server(host: str = "0.0.0.0", port: int = 8001):
    """Start WebSocket server"""
    # Initialize connection manager
    await connection_manager.initialize()
    
    # Start server
    logger.info(f"Starting WebSocket server on {host}:{port}")
    
    async with websockets.serve(websocket_handler, host, port):
        logger.info(f"WebSocket server listening on ws://{host}:{port}")
        await asyncio.Future()  # Run forever