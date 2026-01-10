"""
Database models for conversation storage.
Uses SQLite for persistent storage of chat history.
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional
import json
import uuid


class ConversationDB:
    """
    SQLite database for storing conversations.
    
    Features:
    - Persistent chat history across sessions
    - Multiple conversations per user
    - Message-level storage with metadata
    - Full-text search capability
    """
    
    def __init__(self, db_path: str = "./data/conversations.db"):
        """Initialize the database connection."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with row factory."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn
    
    def _init_db(self):
        """Initialize database tables."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Conversations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                model_used TEXT,
                metadata TEXT
            )
        """)
        
        # Messages table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                audio_path TEXT,
                sources TEXT,
                confidence REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
                    ON DELETE CASCADE
            )
        """)
        
        # Create indexes for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_conversation 
            ON messages(conversation_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_conversations_updated 
            ON conversations(updated_at DESC)
        """)
        
        conn.commit()
        conn.close()
    
    def create_conversation(
        self, 
        title: str = "New Conversation",
        model_used: str = "llama3.2"
    ) -> str:
        """
        Create a new conversation.
        
        Returns:
            Conversation ID
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        conversation_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        
        cursor.execute("""
            INSERT INTO conversations (id, title, created_at, updated_at, model_used)
            VALUES (?, ?, ?, ?, ?)
        """, (conversation_id, title, now, now, model_used))
        
        conn.commit()
        conn.close()
        
        return conversation_id
    
    def get_conversation(self, conversation_id: str) -> Optional[dict]:
        """Get a conversation by ID."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM conversations WHERE id = ?
        """, (conversation_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return dict(row)
        return None
    
    def list_conversations(self, limit: int = 50) -> list[dict]:
        """List all conversations, ordered by most recent."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT c.*, 
                   (SELECT COUNT(*) FROM messages WHERE conversation_id = c.id) as message_count
            FROM conversations c
            ORDER BY c.updated_at DESC
            LIMIT ?
        """, (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def update_conversation_title(self, conversation_id: str, title: str):
        """Update conversation title."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE conversations 
            SET title = ?, updated_at = ?
            WHERE id = ?
        """, (title, datetime.utcnow().isoformat(), conversation_id))
        
        conn.commit()
        conn.close()
    
    def delete_conversation(self, conversation_id: str):
        """Delete a conversation and all its messages."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Messages are deleted via CASCADE
        cursor.execute("""
            DELETE FROM conversations WHERE id = ?
        """, (conversation_id,))
        
        conn.commit()
        conn.close()
    
    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        audio_path: Optional[str] = None,
        sources: Optional[list] = None,
        confidence: Optional[float] = None
    ) -> str:
        """
        Add a message to a conversation.
        
        Args:
            conversation_id: Conversation ID
            role: 'user' or 'assistant'
            content: Message text
            audio_path: Path to audio file (if any)
            sources: Source documents used (for assistant messages)
            confidence: Confidence score (for assistant messages)
            
        Returns:
            Message ID
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        message_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        
        sources_json = json.dumps(sources) if sources else None
        
        cursor.execute("""
            INSERT INTO messages (id, conversation_id, role, content, audio_path, sources, confidence, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (message_id, conversation_id, role, content, audio_path, sources_json, confidence, now))
        
        # Update conversation's updated_at
        cursor.execute("""
            UPDATE conversations SET updated_at = ? WHERE id = ?
        """, (now, conversation_id))
        
        conn.commit()
        conn.close()
        
        return message_id
    
    def get_messages(self, conversation_id: str) -> list[dict]:
        """Get all messages for a conversation."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM messages 
            WHERE conversation_id = ?
            ORDER BY created_at ASC
        """, (conversation_id,))
        
        rows = cursor.fetchall()
        conn.close()
        
        messages = []
        for row in rows:
            msg = dict(row)
            if msg.get("sources"):
                msg["sources"] = json.loads(msg["sources"])
            messages.append(msg)
        
        return messages
    
    def search_conversations(self, query: str, limit: int = 20) -> list[dict]:
        """Search conversations by message content."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT DISTINCT c.*, 
                   (SELECT COUNT(*) FROM messages WHERE conversation_id = c.id) as message_count
            FROM conversations c
            JOIN messages m ON c.id = m.conversation_id
            WHERE m.content LIKE ?
            ORDER BY c.updated_at DESC
            LIMIT ?
        """, (f"%{query}%", limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def get_conversation_with_messages(self, conversation_id: str) -> Optional[dict]:
        """Get a conversation with all its messages."""
        conversation = self.get_conversation(conversation_id)
        if conversation:
            conversation["messages"] = self.get_messages(conversation_id)
        return conversation
    
    def auto_title_conversation(self, conversation_id: str, first_message: str):
        """Auto-generate a title from the first user message."""
        # Take first 50 chars of message as title
        title = first_message[:50].strip()
        if len(first_message) > 50:
            title += "..."
        self.update_conversation_title(conversation_id, title)


# Global database instance
conversation_db = ConversationDB()
