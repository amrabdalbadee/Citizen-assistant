"""
Session Management Service.

Handles conversation state and history for multi-turn interactions:
- Session creation and retrieval
- Conversation history storage
- Session timeout and cleanup
- Memory-efficient storage with TTL

For production, this can be swapped with Redis or a database backend.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional
from uuid import uuid4
from collections import OrderedDict
import threading

from app.core.config import get_settings
from app.models.schemas import ConversationTurn, SessionInfo

logger = logging.getLogger(__name__)
settings = get_settings()


class SessionManager:
    """
    In-memory session manager with TTL support.
    
    Architecture Decisions:
    
    1. In-Memory Storage (Prototype):
       - Fast access for low latency
       - OrderedDict for efficient LRU-like cleanup
       - Thread-safe with locks
    
    2. Production Alternatives:
       - Redis: Distributed caching with built-in TTL
       - PostgreSQL: Persistent storage with query capabilities
       - DynamoDB: Serverless, auto-scaling
    
    3. Session Design:
       - UUID-based session IDs for security
       - Limited history length to prevent memory bloat
       - Automatic cleanup of expired sessions
    
    Scalability Consideration:
    For 1000+ concurrent users, migrate to Redis with:
    - Cluster mode for horizontal scaling
    - Key prefix for namespacing
    - Pub/Sub for real-time updates
    """
    
    _instance: Optional["SessionManager"] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern for global session state."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._sessions: OrderedDict[str, dict] = OrderedDict()
                    cls._instance._cleanup_lock = threading.Lock()
        return cls._instance
    
    def create_session(self) -> str:
        """
        Create a new session.
        
        Returns:
            Unique session ID
        """
        session_id = str(uuid4())
        now = datetime.utcnow()
        
        self._sessions[session_id] = {
            "created_at": now,
            "last_activity": now,
            "history": [],
            "metadata": {}
        }
        
        logger.debug(f"Created session: {session_id}")
        
        # Trigger cleanup periodically
        if len(self._sessions) % 100 == 0:
            self._cleanup_expired()
        
        return session_id
    
    def get_session(self, session_id: str) -> Optional[dict]:
        """
        Retrieve a session by ID.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session data or None if not found/expired
        """
        session = self._sessions.get(session_id)
        
        if session is None:
            return None
        
        # Check expiration
        timeout = timedelta(minutes=settings.session_timeout_minutes)
        if datetime.utcnow() - session["last_activity"] > timeout:
            self.delete_session(session_id)
            return None
        
        return session
    
    def update_activity(self, session_id: str):
        """Update the last activity timestamp for a session."""
        if session_id in self._sessions:
            self._sessions[session_id]["last_activity"] = datetime.utcnow()
            # Move to end (most recent)
            self._sessions.move_to_end(session_id)
    
    def add_turn(
        self,
        session_id: str,
        role: str,
        content: str
    ) -> bool:
        """
        Add a conversation turn to session history.
        
        Args:
            session_id: Session identifier
            role: 'user' or 'assistant'
            content: Message content
            
        Returns:
            True if successful, False if session not found
        """
        session = self.get_session(session_id)
        if session is None:
            return False
        
        turn = {
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        session["history"].append(turn)
        
        # Limit history length
        if len(session["history"]) > settings.max_history_length * 2:
            # Keep only recent turns
            session["history"] = session["history"][-settings.max_history_length * 2:]
        
        self.update_activity(session_id)
        return True
    
    def get_history(self, session_id: str) -> list[dict]:
        """
        Get conversation history for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of conversation turns
        """
        session = self.get_session(session_id)
        if session is None:
            return []
        return session["history"]
    
    def get_session_info(self, session_id: str) -> Optional[SessionInfo]:
        """
        Get detailed session information.
        
        Args:
            session_id: Session identifier
            
        Returns:
            SessionInfo object or None
        """
        session = self.get_session(session_id)
        if session is None:
            return None
        
        history = [
            ConversationTurn(
                role=turn["role"],
                content=turn["content"],
                timestamp=datetime.fromisoformat(turn["timestamp"])
            )
            for turn in session["history"]
        ]
        
        return SessionInfo(
            session_id=session_id,
            created_at=session["created_at"],
            last_activity=session["last_activity"],
            conversation_history=history,
            total_queries=len([t for t in history if t.role == "user"])
        )
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if deleted, False if not found
        """
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.debug(f"Deleted session: {session_id}")
            return True
        return False
    
    def _cleanup_expired(self):
        """Remove expired sessions."""
        with self._cleanup_lock:
            timeout = timedelta(minutes=settings.session_timeout_minutes)
            now = datetime.utcnow()
            
            expired = [
                sid for sid, session in self._sessions.items()
                if now - session["last_activity"] > timeout
            ]
            
            for sid in expired:
                del self._sessions[sid]
            
            if expired:
                logger.info(f"Cleaned up {len(expired)} expired sessions")
    
    def get_active_session_count(self) -> int:
        """Get count of active (non-expired) sessions."""
        self._cleanup_expired()
        return len(self._sessions)
    
    def clear_all(self):
        """Clear all sessions (for testing)."""
        self._sessions.clear()
        logger.info("All sessions cleared")


# Global session manager instance
session_manager = SessionManager()
