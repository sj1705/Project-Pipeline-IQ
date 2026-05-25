import uuid
from datetime import datetime
from sqlalchemy import Column, String, Integer, Float, Text, Boolean, DateTime, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
from app.models.database import Base


class Document(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String(255), nullable=False)
    file_type = Column(String(10), nullable=False)
    s3_path = Column(Text, nullable=True)
    ingested_at = Column(DateTime, default=datetime.utcnow)
    chunk_config = Column(JSON, nullable=True)

    chunks = relationship("Chunk", back_populates="document")

    def __repr__(self):
        return f"<Document {self.filename}>"


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(1024), nullable=True)  # changed from 1536 to 1024
    chunk_index = Column(Integer, nullable=False)
    chunk_size = Column(Integer, nullable=False)
    overlap = Column(Integer, default=0)
    metadata_ = Column("metadata", JSON, nullable=True)

    document = relationship("Document", back_populates="chunks")

    def __repr__(self):
        return f"<Chunk {self.id} of {self.document_id}>"


class QueryLog(Base):
    __tablename__ = "query_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    query = Column(Text, nullable=False)
    response = Column(Text, nullable=True)
    model_used = Column(String(50), nullable=True)
    latency_ms = Column(Float, nullable=True)
    token_count = Column(Integer, nullable=True)
    cost = Column(Float, nullable=True)
    retrieval_scores = Column(JSON, nullable=True)
    evaluation_scores = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<QueryLog {self.id} model={self.model_used}>"


class PipelineConfig(Base):
    __tablename__ = "pipeline_configs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    version = Column(Integer, nullable=False)
    chunk_size = Column(Integer, default=512)
    chunk_overlap = Column(Integer, default=50)
    top_k = Column(Integer, default=5)
    rerank_weight = Column(Float, default=0.5)
    routing_threshold = Column(Float, default=0.5)
    retry_threshold = Column(Float, default=0.7)
    is_active = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<PipelineConfig v{self.version} active={self.is_active}>"

class QueryCache(Base):
    __tablename__ = "query_cache"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    question = Column(Text, nullable=False)
    embedding = Column(Vector(1024), nullable=False)
    response = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<QueryCache {self.question[:50]}>"