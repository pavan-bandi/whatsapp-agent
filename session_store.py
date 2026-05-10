"""
Session Store
In-memory session management for uploaded chat data.
In production, replace with Redis or a database.
"""

import uuid
from datetime import datetime
from typing import Optional

from app.models.schemas import ParsedMessage, SessionInfo, ConversationStats


class SessionStore:
    """
    In-memory store for chat sessions.
    Each session holds parsed messages and computed stats.
    """
    
    def __init__(self):
        self._sessions: dict[str, dict] = {}
    
    def create_session(
        self,
        messages: list[ParsedMessage],
        stats: ConversationStats,
        participants: list[str],
        date_range: dict,
    ) -> str:
        session_id = str(uuid.uuid4())
        self._sessions[session_id] = {
            "session_id": session_id,
            "messages": messages,
            "stats": stats,
            "participants": participants,
            "date_range": date_range,
            "created_at": datetime.utcnow(),
            "is_indexed": False,
        }
        return session_id
    
    def get_session(self, session_id: str) -> Optional[dict]:
        return self._sessions.get(session_id)
    
    def get_messages(self, session_id: str) -> Optional[list[ParsedMessage]]:
        session = self._sessions.get(session_id)
        return session["messages"] if session else None
    
    def get_stats(self, session_id: str) -> Optional[ConversationStats]:
        session = self._sessions.get(session_id)
        return session["stats"] if session else None
    
    def mark_indexed(self, session_id: str):
        if session_id in self._sessions:
            self._sessions[session_id]["is_indexed"] = True
    
    def is_indexed(self, session_id: str) -> bool:
        session = self._sessions.get(session_id)
        return session["is_indexed"] if session else False
    
    def delete_session(self, session_id: str) -> bool:
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False
    
    def list_sessions(self) -> list[SessionInfo]:
        return [
            SessionInfo(
                session_id=s["session_id"],
                participants=s["participants"],
                total_messages=len(s["messages"]),
                date_range=s["date_range"],
                created_at=s["created_at"],
                is_indexed=s["is_indexed"],
            )
            for s in self._sessions.values()
        ]


# Singleton instance
session_store = SessionStore()
