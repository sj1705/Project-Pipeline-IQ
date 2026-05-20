import uuid
from datetime import datetime
from sqlalchemy import Column, String, Integer, Float, Text, Boolean, DateTime, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from app.models.database import Base


class Document(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String(255), nullable=False)
    file_type = Column(String(10), nullable=False)  # pdf, docx, html
    s3_path = Column(Text, nullable=True)
    ingested_at = Column(DateTime, default=datetime.utcnow)
    chunk_config = Column(JSON, nullable=True)  # stores chunking params used

    # Relationship — one document has many chunks
    chunks = relationship("Chunk", back_populates="document")

    def __repr__(self):
        return f"<Document {self.filename}>"


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    content = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    chunk_size = Column(Integer, nullable=False)
    overlap = Column(Integer, default=0)
    metadata_ = Column("metadata", JSON, nullable=True)  # renamed to avoid Python conflict

    # Relationship — each chunk belongs to one document
    document = relationship("Document", back_populates="chunks")

    def __repr__(self):
        return f"<Chunk {self.id} of {self.document_id}>"