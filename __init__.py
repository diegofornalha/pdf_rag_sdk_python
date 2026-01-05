"""PDF RAG SDK - PDF to SQLite Vector Database.

Simplified SDK for converting PDFs to vector embeddings stored in SQLite.

Example:
    >>> from pdf_rag_sdk_python import IngestEngine, ChunkingStrategy
    >>>
    >>> async def main():
    ...     engine = IngestEngine(
    ...         db_path="data/docs.db",
    ...         embedding_model="BAAI/bge-small-en-v1.5",
    ...         chunk_size=350,
    ...         chunk_overlap=70,
    ...         chunking_strategy=ChunkingStrategy.FIXED
    ...     )
    ...     result = await engine.add_document("document.pdf")
    ...     print(f"Created {result.chunks} chunks")
"""

from .ingest import Document, IngestEngine, IngestResult
from .options import ChunkingStrategy, EmbeddingModel

__version__ = "0.2.0-minimal"

__all__ = [
    # Options/Config
    "EmbeddingModel",
    "ChunkingStrategy",
    # Ingest
    "IngestEngine",
    "IngestResult",
    "Document",
]
