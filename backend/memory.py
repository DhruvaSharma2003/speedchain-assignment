# backend/memory.py
from datetime import datetime
from typing import Optional, Dict
from sqlalchemy import create_engine, Column, String, Integer, Text, DateTime, ForeignKey, UniqueConstraint
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

engine = create_engine("sqlite:///memory.db", future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
Base = declarative_base()

class ChatSession(Base):
    __tablename__ = "sessions"
    id = Column(String, primary_key=True)           # client-provided session_id
    created_at = Column(DateTime, default=datetime.utcnow)

    messages = relationship("Message", back_populates="session", cascade="all, delete-orphan")
    entities = relationship("Entity", back_populates="session", cascade="all, delete-orphan")

class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, ForeignKey("sessions.id"), index=True)
    role = Column(String)                           # "user" | "assistant"
    text = Column(Text)
    ts = Column(DateTime, default=datetime.utcnow)

    session = relationship("ChatSession", back_populates="messages")

class Entity(Base):
    __tablename__ = "entities"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, ForeignKey("sessions.id"), index=True)
    key = Column(String)
    value = Column(String)
    __table_args__ = (UniqueConstraint("session_id", "key", name="uq_session_key"),)

    session = relationship("ChatSession", back_populates="entities")

def init_db():
    Base.metadata.create_all(bind=engine)

def get_or_create_session(db, session_id: str) -> ChatSession:
    s = db.get(ChatSession, session_id)
    if not s:
        s = ChatSession(id=session_id)
        db.add(s)
        db.commit()
    return s

def add_message(db, session_id: str, role: str, text: str):
    m = Message(session_id=session_id, role=role, text=text)
    db.add(m)
    db.commit()
    return m

def upsert_entity(db, session_id: str, key: str, value: str):
    e = db.query(Entity).filter_by(session_id=session_id, key=key).one_or_none()
    if e:
        e.value = value
    else:
        e = Entity(session_id=session_id, key=key, value=value)
        db.add(e)
    db.commit()

def get_entities(db, session_id: str) -> Dict[str, str]:
    ents = db.query(Entity).filter_by(session_id=session_id).all()
    return {e.key: e.value for e in ents}
