"""
Database models using SQLAlchemy ORM.
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, JSON
from database.connection import Base


class Chat(Base):
    """Chat session model."""
    
    __tablename__ = "chats"
    
    id = Column(String, primary_key=True)
    title = Column(String, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    sources = Column(JSON, nullable=False, default=list)  # List of search sources
    files = Column(JSON, nullable=False, default=list)  # List of uploaded files
    
    def to_dict(self):
        """Convert to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at.isoformat(),
            "sources": self.sources,
            "files": self.files
        }


class Message(Base):
    """Message model."""
    
    __tablename__ = "messages"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    message_id = Column(String, nullable=False, unique=True)
    chat_id = Column(String, nullable=False)
    backend_id = Column(String, nullable=False)  # Session ID
    query = Column(Text, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    response_blocks = Column(JSON, nullable=False, default=list)  # List of response blocks
    status = Column(
        String,
        nullable=False,
        default="answering"
    )  # answering, completed, error
    
    def to_dict(self):
        """Convert to dictionary."""
        return {
            "id": self.id,
            "message_id": self.message_id,
            "chat_id": self.chat_id,
            "backend_id": self.backend_id,
            "query": self.query,
            "created_at": self.created_at.isoformat(),
            "response_blocks": self.response_blocks,
            "status": self.status
        }
