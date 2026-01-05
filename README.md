# PDF RAG SDK Python

SDK minimalista para converter PDFs em banco de dados vetorial SQLite, otimizado para aplicações RAG (Retrieval-Augmented Generation).

## 🎯 Propósito

Transformar documentos PDF em chunks de texto com embeddings vetoriais, armazenados em SQLite para busca semântica eficiente.

## ✨ Features

- ✅ **Extração de texto de PDF** usando `pypdf`
- ✅ **Chunking inteligente** com 3 estratégias (FIXED, SENTENCE, PARAGRAPH)
- ✅ **Overlap configurável** para preservar contexto entre chunks
- ✅ **Embeddings rápidos** com FastEmbed (CPU-based, sem GPU necessária)
- ✅ **Armazenamento vetorial** em SQLite com extensão `sqlite-vec`
- ✅ **Deduplicação automática** via hash SHA256
- ✅ **Zero dependências complexas** - apenas 4 bibliotecas essenciais

## 📦 Instalação

```bash
pip install -r requirements.txt
```

### Dependências

```txt
apsw>=3.9.0          # SQLite async wrapper
sqlite-vec>=0.1.0    # Extensão vetorial
fastembed>=0.2.0     # Embeddings CPU
pypdf>=3.0.0         # Leitura de PDF
```

## 🚀 Uso Rápido

```python
import asyncio
from pdf_rag_sdk_python import IngestEngine, ChunkingStrategy

async def main():
    # Configurar engine
    engine = IngestEngine(
        db_path="data/documentos.db",
        embedding_model="BAAI/bge-small-en-v1.5",
        chunk_size=350,           # palavras por chunk
        chunk_overlap=70,         # 20% de overlap
        chunking_strategy=ChunkingStrategy.FIXED
    )

    # Ingerir PDF
    result = await engine.add_document("documento.pdf")

    if result.success:
        print(f"✅ {result.chunks} chunks criados")
        print(f"📊 Doc ID: {result.doc_id}")
    else:
        print(f"❌ Erro: {result.error}")

    # Estatísticas
    stats = engine.stats
    print(f"Total documentos: {stats['total_documents']}")
    print(f"Total chunks: {stats['total_chunks']}")

asyncio.run(main())
```

## ⚙️ Configurações

### Estratégias de Chunking

| Estratégia | Descrição | Uso Recomendado |
|------------|-----------|-----------------|
| `FIXED` | Tamanho fixo de palavras | Documentos estruturados (regulamentos, manuais) |
| `SENTENCE` | Quebra por sentença | Textos narrativos |
| `PARAGRAPH` | Quebra por parágrafo | Artigos, documentação |

### Modelos de Embedding Suportados

```python
# Padrão (recomendado)
embedding_model="BAAI/bge-small-en-v1.5"  # 384 dimensões

# Alternativas
embedding_model="BAAI/bge-base-en-v1.5"   # 768 dimensões
embedding_model="BAAI/bge-large-en-v1.5"  # 1024 dimensões
```

## 🗄️ Estrutura do Banco de Dados

### Tabelas Criadas

**`documentos`** - Metadados dos PDFs
```sql
id, nome, tipo, conteudo, caminho, hash, metadata, criado_em
```

**`chunks`** - Texto dividido
```sql
id, doc_id, chunk_index, conteudo
```

**`vec_chunks`** - Embeddings vetoriais
```sql
chunk_id, embedding (busca por similaridade)
```

## 📊 Exemplo de Resultado

```
======================================================================
INGESTÃO DO PDF - REGULAMENTO
======================================================================
📄 PDF: regulamento.pdf
💾 DB:  data/regulamento.db
🧠 Modelo: BAAI/bge-small-en-v1.5
📏 Chunk: 350 palavras | Overlap: 70 (20%)
======================================================================

✅ SUCESSO!
   Doc ID: 1
   Chunks: 59
   
📊 Estatísticas:
   Documentos: 1
   Chunks: 59
   Tamanho: 134,253 bytes
```

## 🔍 Overlap Preservado

```
Chunk 0 → 1: ~94 palavras de overlap
Chunk 1 → 2: ~92 palavras de overlap  
Chunk 2 → 3: ~95 palavras de overlap
```

## 📁 Formatos Suportados

| Formato | Suporte | Biblioteca |
|---------|---------|------------|
| `.pdf` | ✅ Principal | pypdf |
| `.txt` | ✅ Fallback | stdlib |
| `.md` | ✅ Fallback | stdlib |
| `.json` | ✅ Fallback | stdlib |

## 🔧 API Completa

### `add_document(path, metadata=None)`
```python
result = await engine.add_document(
    "documento.pdf",
    metadata={"categoria": "regulamento"}
)
```

### `add_text(text, source, doc_type, metadata=None)`
```python
result = await engine.add_text(
    text="Conteúdo...",
    source="api",
    doc_type="txt"
)
```

### `stats` (property)
```python
stats = engine.stats
# {'total_documents': 10, 'total_chunks': 523, ...}
```

## ⚡ Performance

- **Chunking:** ~1000 palavras/segundo
- **Embedding (CPU):**
  - bge-small: ~100 chunks/seg
  - bge-base: ~50 chunks/seg
  - bge-large: ~25 chunks/seg

## 🛡️ Deduplicação

Hash SHA256 previne duplicatas automaticamente.

## 🚧 Limitações

- ❌ DOCX/HTML não suportados (removidos)
- ❌ Módulo de busca não incluído (apenas ingest)
- ❌ OCR não suportado (PDFs escaneados)

## 📝 Exemplo Completo

Ver: `scripts/ingest_regulamento.py`

## 🔬 Estrutura

```
pdf_rag_sdk_python/
├── __init__.py
├── ingest.py
├── options.py
├── requirements.txt
└── README.md
```

---

**Versão:** 0.2.0-minimal | **Python:** >= 3.10
