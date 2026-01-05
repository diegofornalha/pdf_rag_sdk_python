# PDF RAG SDK Python

SDK simplificado para converter PDFs em embeddings vetoriais armazenados em SQLite.

## Instalação

```bash
pip install -r requirements.txt
```

## Uso

```python
from pdf_rag_sdk_python import IngestEngine, ChunkingStrategy

async def main():
    engine = IngestEngine(
        db_path="data/docs.db",
        embedding_model="BAAI/bge-small-en-v1.5",
        chunk_size=350,
        chunk_overlap=70,
        chunking_strategy=ChunkingStrategy.FIXED
    )
    result = await engine.add_document("document.pdf")
    print(f"Created {result.chunks} chunks")
```

## Dependências

- `apsw>=3.9.0` - SQLite async wrapper
- `sqlite-vec>=0.1.0` - SQLite vector extension
- `fastembed>=0.2.0` - Fast CPU-based embeddings
- `pypdf>=3.0.0` - PDF text extraction

## Versão

0.2.0-minimal

