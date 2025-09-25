# rag.py
# Regenerated: Production-grade local RAG pipeline with OCR, ChromaDB, Sentence-Transformers, and Phi-3-mini fallback.
# - Embeddings: sentence-transformers (all-MiniLM-L6-v2, 384-dim) [local]
# - Vector DB: chromadb (PersistentClient at ./chroma_db)
# - OCR: pdfplumber (PDF), python-docx (DOCX), easyocr/pytesseract (Images)
# - RAG: add_notes_to_db, query_notes (with optional local query expansion)
# - LLM Fallback: microsoft/Phi-3-mini-4k-instruct via transformers (CPU-friendly)
# - Stats: get_database_stats (alias get_db_stats)
# - Robust imports: show st.error in Streamlit context, else raise RuntimeError.

from __future__ import annotations

import os
import hashlib
import shutil
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Optional Streamlit for UI-friendly errors (not required at runtime)
try:
    import streamlit as st  # type: ignore
    _IN_STREAMLIT = True
except Exception:
    st = None  # type: ignore
    _IN_STREAMLIT = False


def _ui_error_or_raise(msg: str) -> None:
    """Show error in Streamlit app if available; otherwise raise RuntimeError."""
    if _IN_STREAMLIT and st is not None:
        st.error(msg)
    else:
        raise RuntimeError(msg)


# ----------------------- Third-party dependencies (graceful) ------------------

# ChromaDB
try:
    import chromadb  # type: ignore
    from chromadb import PersistentClient  # type: ignore
    from chromadb.api.models.Collection import Collection  # type: ignore
except Exception as e:
    _ui_error_or_raise(f"chromadb import failed: {e}. Install with: pip install chromadb")
    raise

# Sentence-Transformers
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception as e:
    _ui_error_or_raise(
        f"sentence-transformers import failed: {e}. Install with: pip install sentence-transformers"
    )
    raise

# Transformers + Torch for Phi-3-mini
try:
    import torch  # type: ignore
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline  # type: ignore
except Exception as e:
    _ui_error_or_raise(
        f"transformers/torch import failed: {e}. Install with: pip install torch transformers"
    )
    raise

# OCR libs (optional)
try:
    import pdfplumber  # type: ignore
except Exception:
    pdfplumber = None  # type: ignore

try:
    import docx  # python-docx
except Exception:
    docx = None  # type: ignore

try:
    import easyocr  # type: ignore
except Exception:
    easyocr = None  # type: ignore

try:
    import pytesseract  # type: ignore
except Exception:
    pytesseract = None  # type: ignore

from shutil import which as _which


# ----------------------------- Configuration ---------------------------------

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # 384-dim embeddings
EMBED_DIM = 384
DEFAULT_PERSIST_DIR = "chroma_db"
COLLECTION_BASE = "study_notes"
COLLECTION_NAME = f"{COLLECTION_BASE}_dim{EMBED_DIM}"  # include dim to avoid mismatches

# Chunking defaults (balanced for recall/latency)
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# Phi-3-mini fallback
PHI_MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"

# Stats scanning
STATS_SCAN_LIMIT = 100_000


# ----------------------------- Module state ----------------------------------

_client: Optional[PersistentClient] = None
_collection: Optional[Collection] = None
_embedder: Optional[SentenceTransformer] = None
_persist_dir: str = DEFAULT_PERSIST_DIR


# ----------------------------- Utilities -------------------------------------

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _ts() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _hash_id(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _normalize_similarity(distance: Optional[float]) -> float:
    """
    Convert cosine distance (0 = best) to similarity in [0, 1]:
      similarity = clamp(1 - distance, 0, 1)
    """
    if distance is None:
        return 0.0
    try:
        sim = 1.0 - float(distance)
        return max(0.0, min(1.0, sim))
    except Exception:
        return 0.0


# ----------------------------- Lazy loaders ----------------------------------

@lru_cache(maxsize=1)
def _get_embedder() -> SentenceTransformer:
    """Load and cache the sentence-transformers model (normalized embeddings)."""
    try:
        model = SentenceTransformer(EMBED_MODEL_NAME)
        return model
    except Exception as e:
        _ui_error_or_raise(
            f"Failed to load embeddings '{EMBED_MODEL_NAME}': {e}. "
            f"Try: pip install sentence-transformers"
        )
        raise


@lru_cache(maxsize=1)
def _get_client(persist_dir: str) -> PersistentClient:
    """Get/Create persistent Chroma client at provided directory."""
    try:
        _ensure_dir(persist_dir)
        return chromadb.PersistentClient(path=persist_dir)
    except Exception as e:
        _ui_error_or_raise(f"Failed to initialize Chroma PersistentClient: {e}")
        raise


@lru_cache(maxsize=1)
def _get_collection_cached(persist_dir: str) -> Collection:
    """Get or create the Chroma collection (name encodes embedding dim)."""
    client = _get_client(persist_dir)
    try:
        col = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"embedding_model": EMBED_MODEL_NAME, "embedding_dim": EMBED_DIM, "hnsw:space": "cosine"},
        )
        return col
    except Exception as e:
        _ui_error_or_raise(f"Failed to get_or_create collection '{COLLECTION_NAME}': {e}")
        raise


@lru_cache(maxsize=1)
def _get_phi_pipe():
    """Lazy-load Phi-3-mini instruct pipeline (CPU)."""
    try:
        tok = AutoTokenizer.from_pretrained(PHI_MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(PHI_MODEL_ID, torch_dtype=torch.float32)
        return pipeline("text-generation", model=model, tokenizer=tok, device=-1)
    except Exception as e:
        _ui_error_or_raise(f"Failed to load Phi-3-mini fallback: {e}")
        raise


# ----------------------------- Initialization --------------------------------

def initialize_rag_system(persist_dir: str = DEFAULT_PERSIST_DIR) -> Tuple[Collection, SentenceTransformer]:
    """
    Initialize and return (collection, embedder).
    Safe to call multiple times; uses caching for speed.
    """
    global _client, _collection, _embedder, _persist_dir
    _persist_dir = persist_dir or DEFAULT_PERSIST_DIR

    # Clear caches if persist dir changed
    _get_client.cache_clear()          # type: ignore
    _get_collection_cached.cache_clear()  # type: ignore

    _embedder = _get_embedder()
    _client = _get_client(_persist_dir)
    _collection = _get_collection_cached(_persist_dir)
    return _collection, _embedder


# ----------------------------- OCR / Extraction ------------------------------

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF using pdfplumber."""
    if pdfplumber is None:
        _ui_error_or_raise("pdfplumber not available. Install with: pip install pdfplumber")
    text_parts: List[str] = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            try:
                t = page.extract_text() or ""
            except Exception:
                t = ""
            if t:
                text_parts.append(t)
    return "\n".join(text_parts).strip()


def extract_text_from_docx(file_path: str) -> str:
    """Extract text from DOCX using python-docx."""
    if docx is None:
        _ui_error_or_raise("python-docx not available. Install with: pip install python-docx")
    d = docx.Document(file_path)  # type: ignore
    return "\n".join(p.text for p in d.paragraphs).strip()


def extract_text_from_image(file_path: str, lang: str = "en") -> str:
    """Extract text from images using EasyOCR if available, otherwise pytesseract (requires Tesseract binary)."""
    # Prefer EasyOCR (no system binary)
    if easyocr is not None:
        try:
            reader = easyocr.Reader([lang], gpu=False)
            results = reader.readtext(file_path, detail=0, paragraph=True)
            return "\n".join(results).strip()
        except Exception:
            pass
    # Fallback: pytesseract if tesseract is installed
    if pytesseract is not None and _which("tesseract"):
        try:
            from PIL import Image  # lazy import
            img = Image.open(file_path)
            return pytesseract.image_to_string(img)  # type: ignore
        except Exception:
            return ""
    return ""


def extract_text(file_path: str) -> str:
    """
    Unified text extraction dispatcher.
    Supports PDF, DOCX, TXT, PNG/JPG/JPEG/TIFF.
    """
    suffix = Path(file_path).suffix.lower()
    if suffix == ".pdf":
        return extract_text_from_pdf(file_path)
    if suffix == ".docx":
        return extract_text_from_docx(file_path)
    if suffix == ".txt":
        try:
            return Path(file_path).read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return ""
    if suffix in {".png", ".jpg", ".jpeg", ".tiff"}:
        return extract_text_from_image(file_path)

    # Fallback raw read
    try:
        return Path(file_path).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


# ----------------------------- Chunking --------------------------------------

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore
except Exception:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore
    except Exception:
        RecursiveCharacterTextSplitter = None  # type: ignore


def _get_splitter() -> "RecursiveCharacterTextSplitter":
    """Create a text splitter with sensible defaults."""
    if RecursiveCharacterTextSplitter is None:
        _ui_error_or_raise(
            "LangChain text splitters not available. Install langchain or langchain-text-splitters."
        )
    return RecursiveCharacterTextSplitter(  # type: ignore
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )


# ----------------------------- Embeddings ------------------------------------

def _compute_embeddings(texts: List[str]) -> List[List[float]]:
    """Encode texts to normalized 384-d embeddings."""
    model = _get_embedder()
    vecs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return [v.tolist() for v in vecs]


# ----------------------------- RAG: Ingest -----------------------------------

def add_notes_to_db(text: str, source_name: str, subject: Optional[str] = None) -> Dict[str, Any]:
    """
    Split text into chunks, embed, and store in Chroma with metadata.
    Returns: {'chunks': int, 'total_words': int, 'source_name': str, 'collection': str}
    """
    if not isinstance(text, str) or not text.strip():
        return {"chunks": 0, "total_words": 0, "source_name": source_name, "collection": COLLECTION_NAME}

    initialize_rag_system(_persist_dir)

    splitter = _get_splitter()
    chunks = splitter.split_text(text) if splitter else [text]
    if not chunks:
        return {"chunks": 0, "total_words": 0, "source_name": source_name, "collection": COLLECTION_NAME}

    ts = _ts()
    ids: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    total_words = 0
    for idx, ch in enumerate(chunks):
        wc = len(ch.split())
        total_words += wc
        ids.append(_hash_id(f"{source_name}|{ts}|{idx}"))
        metadatas.append(
            {
                "source": source_name,
                "subject": subject or "General",
                "chunk_id": idx,
                "word_count": wc,
                "timestamp": ts,
            }
        )

    embeddings = _compute_embeddings(chunks)
    try:
        _get_collection_cached(_persist_dir).add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )
    except Exception as e:
        _ui_error_or_raise(f"Failed to add notes to Chroma: {e}")
        raise

    return {
        "chunks": len(chunks),
        "total_words": total_words,
        "source_name": source_name,
        "collection": COLLECTION_NAME,
    }


# ----------------------------- RAG: Query ------------------------------------

def _expand_query(query: str) -> List[str]:
    """Compact local expansions for better recall; de-duped and capped."""
    q = query.strip()
    if not q:
        return []
    lex = {
        "define": ["definition", "meaning"],
        "explain": ["explanation", "clarify"],
        "advantages": ["benefits", "pros"],
        "disadvantages": ["cons", "limitations"],
        "example": ["instance", "illustration"],
        "dbms": ["database", "sql"],
        "ai": ["artificial intelligence", "ml"],
        "normalization": ["1NF", "2NF", "3NF"],
    }
    expansions = [q]
    ql = q.lower()
    for key, syns in lex.items():
        if key in ql:
            expansions.extend([f"{q} {s}" for s in syns[:2]])
    # Deduplicate
    seen = set()
    uniq: List[str] = []
    for s in expansions:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq[:5]


def ask_phi(query: str, context: str = "") -> str:
    """
    Phi-3-mini answer with optional context.
    Keeps generation small for latency on CPU.
    """
    pipe = _get_phi_pipe()
    if context.strip():
        prompt = (
            "Answer the question concisely using the provided context if helpful.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{query}\n\n"
            "Answer:"
        )
    else:
        prompt = f"Answer the question concisely.\n\nQuestion:\n{query}\n\nAnswer:"
    out = pipe(
        prompt,
        max_new_tokens=256,
        do_sample=False,
        temperature=0.2,
        top_p=0.9,
        repetition_penalty=1.05,
        pad_token_id=pipe.tokenizer.eos_token_id,  # type: ignore
        eos_token_id=pipe.tokenizer.eos_token_id,  # type: ignore
    )
    text = out[0]["generated_text"]
    if "Answer:" in text:
        return text.split("Answer:", 1)[-1].strip()
    return text.strip()


def query_notes(
    query: str,
    k: int = 3,
    subject: Optional[str] = None,
    use_expansion: bool = False,
    use_rerank: bool = False,  # kept for compatibility (not used)
) -> List[Dict[str, Any]]:
    """
    Retrieve top-k chunks from Chroma; fallback to Phi if none.
    Returns list of dicts with: chunk_id, similarity, rerank_score, source, subject, text_snippet, metadata
    """
    if not isinstance(query, str) or not query.strip():
        return []

    initialize_rag_system(_persist_dir)

    where = {"subject": subject} if subject else None
    queries = _expand_query(query) if use_expansion else [query.strip()]
    q_embs = _compute_embeddings(queries)

    try:
        res = _get_collection_cached(_persist_dir).query(
            query_embeddings=q_embs,
            n_results=max(k, 5 if use_expansion else k),
            where=where,
        )
    except Exception as e:
        _ui_error_or_raise(f"Query failed: {e}")
        raise

    ids_batches = res.get("ids") or []
    docs_batches = res.get("documents") or []
    metas_batches = res.get("metadatas") or []
    dists_batches = res.get("distances") or []

    pool: Dict[str, Dict[str, Any]] = {}
    for bi in range(len(ids_batches)):
        ids = ids_batches[bi]
        docs = docs_batches[bi] if bi < len(docs_batches) else []
        metas = metas_batches[bi] if bi < len(metas_batches) else []
        dists = dists_batches[bi] if bi < len(dists_batches) else []
        for j, _id in enumerate(ids):
            doc = docs[j] if j < len(docs) else ""
            meta = metas[j] if j < len(metas) else {}
            dist = dists[j] if j < len(dists) else None
            sim = _normalize_similarity(dist)
            prev = pool.get(_id)
            if prev is None or sim > prev.get("similarity", 0.0):
                pool[_id] = {"id": _id, "text": doc, "metadata": meta, "similarity": sim}

    ranked = sorted(pool.values(), key=lambda x: x.get("similarity", 0.0), reverse=True)[:k]

    results: List[Dict[str, Any]] = []
    for r in ranked:
        meta = r.get("metadata") or {}
        results.append(
            {
                "chunk_id": meta.get("chunk_id"),
                "similarity": r.get("similarity", 0.0),
                "rerank_score": None,
                "source": meta.get("source", "Unknown"),
                "subject": meta.get("subject", "General"),
                "text_snippet": (r.get("text") or "")[:1000],
                "metadata": meta,
            }
        )

    # Fallback: Phi direct answer
    if not results:
        try:
            answer = ask_phi(query, context="")
            results.append(
                {
                    "chunk_id": None,
                    "similarity": 0.0,
                    "rerank_score": None,
                    "source": "phi-3-mini",
                    "subject": subject or "General",
                    "text_snippet": answer,
                    "metadata": {"generator": "phi-3-mini-fallback"},
                }
            )
        except Exception:
            return []

    return results


# ----------------------------- Stats & Reset ---------------------------------

def get_database_stats() -> Dict[str, Any]:
    """
    Return DB stats: collection, embedding model/dim, chunks, total_words (approx), sources, subjects list.
    """
    initialize_rag_system(_persist_dir)

    stats: Dict[str, Any] = {
        "collection": COLLECTION_NAME,
        "embedding_model": EMBED_MODEL_NAME,
        "embedding_dim": EMBED_DIM,
        "chunks": 0,
        "total_words": 0,
        "sources": 0,
        "subjects": [],
    }

    # Count chunks
    try:
        stats["chunks"] = int(_get_collection_cached(_persist_dir).count())
    except Exception:
        stats["chunks"] = 0

    # Estimate words/sources/subjects via metadata scan
    words = 0
    sources: set[str] = set()
    subjects: set[str] = set()
    try:
        limit = 5000
        fetched = 0
        # Some Chroma versions support offset pagination
        while fetched < min(STATS_SCAN_LIMIT, max(stats["chunks"], STATS_SCAN_LIMIT)):
            batch = _get_collection_cached(_persist_dir).get(include=["metadatas"], limit=limit, offset=fetched)
            metas = batch.get("metadatas") or []
            if not metas:
                break
            flat = metas[0] if len(metas) == 1 and isinstance(metas[0], list) else metas
            if not flat:
                break
            for m in flat:
                if not isinstance(m, dict):
                    continue
                wc = int(m.get("word_count", 0) or 0)
                words += wc
                src = m.get("source")
                if src:
                    sources.add(str(src))
                sbj = m.get("subject")
                if sbj:
                    subjects.add(str(sbj))
            fetched += len(flat)
            if len(flat) < limit:
                break
    except Exception:
        pass

    stats["total_words"] = int(words)
    stats["sources"] = len(sources)
    stats["subjects"] = sorted(subjects)
    return stats


def get_db_stats() -> Dict[str, Any]:
    """Alias for get_database_stats (for compatibility)."""
    return get_database_stats()


def reset_db(persist_dir: str = DEFAULT_PERSIST_DIR) -> None:
    """Delete persistent DB directory and reinitialize clean."""
    try:
        if os.path.isdir(persist_dir):
            shutil.rmtree(persist_dir)
        # Clear caches/state
        _get_client.cache_clear()             # type: ignore
        _get_collection_cached.cache_clear()  # type: ignore
        initialize_rag_system(persist_dir)
    except Exception as e:
        _ui_error_or_raise(f"Failed to reset DB: {e}")
        raise
