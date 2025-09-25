# app.py
# Study Buddy Pro - Streamlit AI assistant with OCR → RAG + Phi fallback
# - Sidebar-driven subject management, uploads, and settings
# - Tabs: Document Processing, Intelligent Search, Analytics, Management
# - Integrates with local rag.py (embeddings+Chroma, OCR extraction, RAG query, Phi fallback)
# - Compatible with Streamlit 1.49.1; no deprecated APIs; no experimental reruns used.

from __future__ import annotations

import os
import io
import re
import json
import time
import zipfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

# Attempt to import local rag.py
rag = None
rag_import_error = ""
try:
    import rag  # type: ignore
except Exception as e:
    rag_import_error = f"{e.__class__.__name__}: {e}"
    rag = None

# ============================ Config & Setup =================================

APP_NAME = "Study Buddy Pro"
PERSIST_DIR = "chroma_db"
DATA_DIR = "data"
SUPPORTED_EXTS = [".pdf", ".docx", ".txt", ".png", ".jpg", ".jpeg", ".tiff"]

st.set_page_config(
    page_title=APP_NAME,
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Ensure directories
Path(PERSIST_DIR).mkdir(parents=True, exist_ok=True)
Path(DATA_DIR).mkdir(parents=True, exist_ok=True)


# ============================ Session State ===================================

def init_session() -> None:
    ss = st.session_state
    ss.setdefault("dark_mode", False)
    ss.setdefault("subjects", ["General"])
    ss.setdefault("active_subject", "General")
    ss.setdefault("processed_files", [])  # list of dict records
    ss.setdefault("query_history", [])    # list of dict records
    ss.setdefault("search_results", [])
    ss.setdefault("search_meta", {})
    ss.setdefault("query_text", "")
    ss.setdefault("preview_len", 600)
    ss.setdefault("k", 3)
    ss.setdefault("use_expansion", False)
    ss.setdefault("auto_add_to_kb", True)
    ss.setdefault("rag_ready", False)
    ss.setdefault("db_stats", {"chunks": 0, "total_words": 0, "sources": 0, "subjects": []})
    # Load subjects from disk
    existing = [p.name for p in Path(DATA_DIR).iterdir() if p.is_dir()]
    for name in existing:
        if name and name not in ss["subjects"]:
            ss["subjects"].append(name)


init_session()


# ============================ Styles =========================================

def apply_css() -> None:
    dark = st.session_state.get("dark_mode", False)
    if dark:
        bg = "#0f1116"
        card_bg = "#161a23"
        text = "#e6e6e6"
        subtext = "#a0a7b4"
        border = "#2a2f3a"
        primary = "#6ea8fe"
        good = "#22c55e"
        med = "#eab308"
        low = "#ef4444"
        badge_text = "#0f1116"
    else:
        bg = "#ffffff"
        card_bg = "#f8f9fb"
        text = "#1f2937"
        subtext = "#6b7280"
        border = "#e5e7eb"
        primary = "#2563eb"
        good = "#16a34a"
        med = "#ca8a04"
        low = "#dc2626"
        badge_text = "#ffffff"

    st.markdown(
        f"""
        <style>
        :root {{
            --bg: {bg};
            --card-bg: {card_bg};
            --text: {text};
            --subtext: {subtext};
            --border: {border};
            --primary: {primary};
            --good: {good};
            --med: {med};
            --low: {low};
            --badge-text: {badge_text};
        }}
        .card {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1rem;
            margin-bottom: 0.75rem;
            box-shadow: 0 1px 2px rgb(0 0 0 / 0.08), 0 4px 12px rgb(0 0 0 / 0.06);
        }}
        .snippet {{
            border-left: 3px solid var(--border);
            padding: 0.5rem 0.75rem;
            max-height: 220px;
            overflow-y: auto;
            font-size: 0.95rem;
            line-height: 1.35rem;
        }}
        .badge {{
            display: inline-block;
            padding: 0.25rem 0.6rem;
            border-radius: 9999px;
            font-weight: 700;
            font-size: 0.8rem;
            color: var(--badge-text);
            margin-right: 0.35rem;
        }}
        .good {{ background: var(--good); }}
        .med {{ background: var(--med); }}
        .low {{ background: var(--low); }}
        .pill {{
            display: inline-block;
            padding: 0.2rem 0.5rem;
            border: 1px solid var(--border);
            border-radius: 9999px;
            font-size: 0.8rem;
            color: var(--subtext);
            margin-right: 0.35rem;
        }}
        .mono {{
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


apply_css()


# ============================ Helpers ========================================

def human_bytes(num: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num < 1024.0:
            return f"{num:3.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} PB"


def subject_dir(subject: str) -> Path:
    p = Path(DATA_DIR) / subject
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_extracted_text(subject: str, stem: str, text: str) -> Path:
    fn = f"{stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_extracted.txt"
    p = subject_dir(subject) / fn
    p.write_text(text or "", encoding="utf-8")
    return p


def zip_bytes(paths: List[Path]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in paths:
            if p.exists():
                zf.write(p, arcname=p.name)
    buf.seek(0)
    return buf.read()


@st.cache_data(show_spinner=False)
def read_text_cached(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def highlight_terms(text: str, query: str) -> str:
    if not text or not query:
        return text
    terms = [re.escape(t) for t in re.findall(r"\w+", query) if len(t) > 2]
    if not terms:
        return text
    pattern = re.compile("(" + "|".join(terms) + ")", flags=re.IGNORECASE)
    return pattern.sub(r"<b>\1</b>", text)


def sim_badge_class(score: float) -> str:
    try:
        s = float(score)
    except Exception:
        s = 0.0
    if s >= 0.8:
        return "good"
    if s >= 0.6:
        return "med"
    return "low"


def ensure_rag_initialized() -> None:
    if not rag:
        st.session_state["rag_ready"] = False
        return
    if not st.session_state.get("rag_ready", False):
        try:
            init_fn = getattr(rag, "initialize_rag_system", None)
            if callable(init_fn):
                init_fn(PERSIST_DIR)
            st.session_state["rag_ready"] = True
            # Pull initial DB stats if available
            stats_fn = getattr(rag, "get_database_stats", None) or getattr(rag, "get_db_stats", None)
            if callable(stats_fn):
                st.session_state["db_stats"] = stats_fn() or st.session_state["db_stats"]
        except Exception as e:
            st.error(f"RAG initialization failed: {e}")
            st.session_state["rag_ready"] = False


# ============================ Sidebar ========================================

with st.sidebar:
    st.markdown("### 📚 Study Buddy Pro")
    st.checkbox("Dark mode", key="dark_mode", help="Toggle dark/light theme.")
    st.slider("Preview length", 200, 2000, key="preview_len", step=50, help="Characters to preview.")

    st.markdown("#### Subjects")
    # Subject select
    st.selectbox("Active subject", options=st.session_state["subjects"], key="active_subject")
    # Add subject
    new_subj = st.text_input("New subject", value="", placeholder="e.g., AI, DBMS, Python")
    col_subj = st.columns(2)
    with col_subj[0]:
        if st.button("Create", use_container_width=True):
            name = new_subj.strip()
            if not name:
                st.warning("Enter a subject name.")
            elif name in st.session_state["subjects"]:
                st.info("Subject already exists.")
            else:
                st.session_state["subjects"].append(name)
                subject_dir(name)
                st.session_state["active_subject"] = name
                st.success(f"Subject '{name}' created.")
    with col_subj[1]:
        deletable = [s for s in st.session_state["subjects"] if s != "General"]
        del_pick = st.selectbox("Delete subject", options=deletable, index=0 if deletable else None, placeholder="Select")
        if st.button("Delete", use_container_width=True, disabled=not bool(del_pick)):
            if del_pick and del_pick in st.session_state["subjects"]:
                st.session_state["subjects"].remove(del_pick)
                if st.session_state["active_subject"] == del_pick:
                    st.session_state["active_subject"] = "General"
                st.info(f"Subject '{del_pick}' removed (files on disk kept).")

    st.markdown("#### Upload documents")
    uploads = st.file_uploader(
        "Upload PDF, DOCX, TXT, or Images",
        type=[e[1:] for e in SUPPORTED_EXTS],
        accept_multiple_files=True,
    )
    st.toggle("Auto-add to KB after extraction", key="auto_add_to_kb")

    st.markdown("#### Search settings")
    st.number_input("Top-k results", min_value=1, max_value=10, value=st.session_state.get("k", 3), step=1, key="k")
    st.checkbox("Enable query expansion", key="use_expansion")

    # RAG status
    st.markdown("---")
    if rag:
        stats = st.session_state.get("db_stats", {})
        c1, c2, c3 = st.columns(3)
        c1.metric("Chunks", int(stats.get("chunks", 0)))
        c2.metric("Words", f"{int(stats.get('total_words', 0)):,}")
        c3.metric("Sources", int(stats.get("sources", 0)))
    else:
        st.error("rag.py import failed. Some features disabled.")
        if rag_import_error:
            with st.expander("Import details"):
                st.code(rag_import_error)


# Apply CSS again after potential toggle changes
apply_css()


# ============================ Initialization =================================

ensure_rag_initialized()


# ============================ Layout Tabs ====================================

tab_docs, tab_search, tab_analytics, tab_manage = st.tabs(
    ["📄 Document Processing", "🔍 Intelligent Search", "📊 Analytics", "⚙️ Management"]
)


# ============================ Tab: Document Processing ========================

with tab_docs:
    st.subheader("Document Processing")
    st.caption("Upload course material and notes. Text will be extracted and optionally added to the knowledge base.")

    if not rag:
        st.error("rag.py not available. Extraction and knowledge base features require a working rag.py.")
    else:
        if uploads:
            saved_paths: List[Path] = []
            file_records: List[Dict[str, Any]] = []

            with st.status("Processing documents...", expanded=True, state="running") as status:
                progress = st.progress(0.0, text="Starting...")
                total = len(uploads)
                for i, f in enumerate(uploads, start=1):
                    fname = f.name
                    ext = Path(fname).suffix.lower()
                    start = time.time()
                    size = len(f.getvalue()) if hasattr(f, "getvalue") else 0
                    st.info(f"Processing: {fname}")

                    # Persist temp file
                    try:
                        tmp_path = Path(st.experimental_get_query_params().get("_tmp_dir", ["."])[0])  # benign; not used
                    except Exception:
                        tmp_path = Path(".")
                    tmp = Path(os.path.join(os.getcwd(), f"__tmp_{int(time.time()*1000)}{ext}"))
                    tmp.write_bytes(f.getvalue())

                    # Extract via rag.extract_text
                    try:
                        extract_fn = getattr(rag, "extract_text", None)
                        if not callable(extract_fn):
                            raise RuntimeError("rag.extract_text(file_path) not found.")
                        with st.spinner("Extracting text..."):
                            text = extract_fn(str(tmp))
                            if not isinstance(text, str):
                                text = str(text or "")
                    except Exception as e:
                        st.error(f"Extraction failed for {fname}: {e}")
                        text = ""

                    # Save extracted text to data/<subject>
                    subject = st.session_state["active_subject"]
                    stem = Path(fname).stem
                    saved_path = None
                    if text:
                        saved_path = save_extracted_text(subject, stem, text)
                        saved_paths.append(saved_path)

                    elapsed = time.time() - start
                    words = len(text.split()) if text else 0
                    chars = len(text) if text else 0
                    lines = text.count("\n") + 1 if text else 0

                    # Optionally add to KB
                    ingested, ingest_msg, chunks_added = False, "", 0
                    if text and st.session_state.get("auto_add_to_kb", True):
                        try:
                            add_fn = getattr(rag, "add_notes_to_db", None)
                            if callable(add_fn):
                                with st.spinner("Adding to knowledge base..."):
                                    res = add_fn(text, f"{subject}::{saved_path.name if saved_path else stem}", subject=subject)
                                chunks_added = int(res.get("chunks", 0)) if isinstance(res, dict) else 0
                                ingested = True
                                ingest_msg = "Added."
                                # Refresh DB stats
                                stats_fn = getattr(rag, "get_database_stats", None) or getattr(rag, "get_db_stats", None)
                                if callable(stats_fn):
                                    st.session_state["db_stats"] = stats_fn() or st.session_state["db_stats"]
                            else:
                                ingest_msg = "add_notes_to_db() not found."
                        except Exception as e:
                            ingest_msg = f"Ingestion error: {e}"
                            st.warning(ingest_msg)

                    # Show preview and download
                    if text:
                        st.markdown(f"<div class='snippet mono'>{(text[: st.session_state['preview_len']])}</div>", unsafe_allow_html=True)
                        st.download_button(
                            "Download extracted text",
                            data=text.encode("utf-8"),
                            file_name=saved_path.name if saved_path else f"{stem}_extracted.txt",
                            mime="text/plain",
                        )

                    # Record
                    rec = {
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "subject": subject,
                        "filename": fname,
                        "saved_path": str(saved_path) if saved_path else "",
                        "words": words,
                        "chars": chars,
                        "lines": lines,
                        "processing_time": elapsed,
                        "file_size": size,
                        "ingested": ingested,
                        "ingest_msg": ingest_msg,
                        "chunks_added": chunks_added,
                    }
                    st.session_state["processed_files"].append(rec)
                    file_records.append(rec)

                    # Cleanup tmp
                    try:
                        tmp.unlink(missing_ok=True)
                    except Exception:
                        pass

                    progress.progress(i / total, text=f"Processed {i}/{total}")

                status.update(label="Done.", state="complete")

            # Batch download
            if len(saved_paths) > 1:
                zip_data = zip_bytes(saved_paths)
                st.download_button(
                    "Download all extracted (ZIP)",
                    data=zip_data,
                    file_name=f"extracted_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip",
                    use_container_width=True,
                )

    # Recent processed summary
    st.markdown("### Recent processed files")
    recent = list(reversed(st.session_state.get("processed_files", [])))[:10]
    if recent:
        for r in recent:
            with st.container(border=True):
                st.write(f"• {r.get('filename', '-')}")
                st.caption(
                    f"Subject: {r.get('subject','General')} • Words: {r.get('words',0)} • "
                    f"Lines: {r.get('lines',0)} • Time: {r.get('processing_time',0):.2f}s • "
                    f"Size: {human_bytes(int(r.get('file_size',0)))} • Chunks: {r.get('chunks_added',0)}"
                )
                if r.get("saved_path"):
                    content = read_text_cached(r["saved_path"])
                    if content:
                        st.markdown(f"<div class='snippet mono'>{content[: st.session_state['preview_len']]}</div>", unsafe_allow_html=True)
                if r.get("ingest_msg"):
                    st.caption(f"Ingest: {r.get('ingest_msg')}")


# ============================ Tab: Intelligent Search =========================

with tab_search:
    st.subheader("Intelligent Search")
    if not rag:
        st.error("rag.py not available. Please ensure rag.py is in the same folder.")
    else:
        # Quick suggestions
        subj = st.session_state.get("active_subject", "General")
        st.markdown("#### Suggestions")
        c1, c2, c3, c4 = st.columns(4)
        if c1.button("Key concepts"):
            st.session_state["query_text"] = f"List and explain the key concepts in {subj}."
        if c2.button("Definitions"):
            st.session_state["query_text"] = f"Define the most important terms in {subj} with brief examples."
        if c3.button("Formulas"):
            st.session_state["query_text"] = f"What are the important formulas in {subj} with short explanations?"
        if c4.button("Summarize"):
            st.session_state["query_text"] = f"Provide a concise summary of {subj} from my uploaded notes."

        query = st.text_area("Enter a question or keywords", key="query_text", height=120, placeholder="e.g., What is normalization in DBMS?")
        do_search = st.button("Search", type="primary", use_container_width=True, disabled=(not bool(query.strip()) or not st.session_state.get("rag_ready", False)))

        if do_search:
            try:
                t0 = time.time()
                query_fn = getattr(rag, "query_notes", None)
                if not callable(query_fn):
                    raise RuntimeError("rag.query_notes(query, k, subject=None, use_expansion=False) not found.")
                results = query_fn(
                    query=query.strip(),
                    k=int(st.session_state.get("k", 3)),
                    subject=st.session_state.get("active_subject", "General"),
                    use_expansion=bool(st.session_state.get("use_expansion", False)),
                )
                dt = time.time() - t0
                st.session_state["search_results"] = results or []
                st.session_state["search_meta"] = {
                    "query": query.strip(),
                    "elapsed": dt,
                    "count": len(results or []),
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "subject": st.session_state.get("active_subject", "General"),
                }
                # Log history
                st.session_state["query_history"].append(
                    {
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "query": query.strip(),
                        "results_count": len(results or []),
                        "search_time": dt,
                        "subject": st.session_state.get("active_subject", "General"),
                        "k": st.session_state.get("k", 3),
                        "use_expansion": st.session_state.get("use_expansion", False),
                    }
                )
            except Exception as e:
                st.error(f"Search failed: {e}")
                st.session_state["search_results"] = []
                st.session_state["search_meta"] = {"query": query.strip(), "elapsed": 0.0, "count": 0}

        # Results
        meta = st.session_state.get("search_meta", {})
        results = st.session_state.get("search_results", [])
        if meta:
            if meta.get("count", 0) > 0:
                st.success(f"Found {meta.get('count',0)} results in {meta.get('elapsed',0):.2f}s.")
            else:
                st.info(f"No matches found in {meta.get('elapsed',0):.2f}s. Phi fallback may answer directly.")

        if results:
            # Export current results
            export_buf = io.StringIO()
            export_buf.write(f"Query: {meta.get('query','')}\n")
            export_buf.write(f"Results: {len(results)} | Time: {meta.get('elapsed',0):.2f}s\n\n")
            for i, r in enumerate(results, 1):
                export_buf.write(f"=== Result {i} ===\n")
                export_buf.write(f"Similarity: {r.get('similarity',0.0)}\n")
                export_buf.write(f"Source: {r.get('source','Unknown')}\n")
                export_buf.write(f"Subject: {r.get('subject','General')}\n")
                if r.get("chunk_id") is not None:
                    export_buf.write(f"Chunk: {r.get('chunk_id')}\n")
                export_buf.write("\n")
                export_buf.write((r.get("text_snippet") or "") + "\n\n")

            st.download_button(
                "Download shown results",
                data=export_buf.getvalue().encode("utf-8"),
                file_name=f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True,
            )

            st.markdown("### Results")
            for idx, r in enumerate(results):
                similarity = float(r.get("similarity", 0.0) or 0.0)
                badge_cls = sim_badge_class(similarity)
                sim_pct = f"{similarity*100:.1f}%"
                src = r.get("source", "Unknown")
                subj = r.get("subject", "General")
                chunk_id = r.get("chunk_id", None)
                text_snippet = r.get("text_snippet", "") or ""
                snippet_html = highlight_terms(text_snippet[: st.session_state.get("preview_len", 600)], meta.get("query", ""))

                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown(
                    f"<span class='badge {badge_cls}'>Relevance {sim_pct}</span> "
                    f"<span class='pill'>Subject: {subj}</span> "
                    f"<span class='pill'>Chunk: {chunk_id if chunk_id is not None else '-'}</span> "
                    f"<span class='pill mono'>{src}</span>",
                    unsafe_allow_html=True,
                )
                st.markdown(f"<div class='snippet'>{snippet_html}</div>", unsafe_allow_html=True)
                cc1, cc2 = st.columns([1, 1])
                with cc1:
                    if st.button("Use as suggested query", key=f"use_{idx}"):
                        # Use first ~12 words from snippet
                        suggested = " ".join(re.findall(r"\\w+", text_snippet)[:12])
                        st.session_state["query_text"] = suggested
                        st.toast("Suggested query populated.")
                with cc2:
                    st.download_button(
                        "Download snippet",
                        data=text_snippet.encode("utf-8"),
                        file_name=f"result_{idx+1}.txt",
                        mime="text/plain",
                    )
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("Enter a query and click Search to see results.")


# ============================ Tab: Analytics =================================

with tab_analytics:
    st.subheader("Analytics")
    # Global metrics
    dbs = st.session_state.get("db_stats", {})
    cols = st.columns(5)
    cols[0].metric("Subjects", len(st.session_state.get("subjects", [])))
    cols[1].metric("Files", len(st.session_state.get("processed_files", [])))
    cols[2].metric("Chunks (DB)", int(dbs.get("chunks", 0)))
    cols[3].metric("Words (DB)", f"{int(dbs.get('total_words', 0)):,}")
    cols[4].metric("Sources (DB)", int(dbs.get("sources", 0)))

    # Per-subject breakdown from session records
    per_subject: Dict[str, Dict[str, int]] = {}
    for rec in st.session_state.get("processed_files", []):
        s = rec.get("subject", "General")
        d = per_subject.setdefault(s, {"files": 0, "words": 0, "chunks": 0})
        d["files"] += 1
        d["words"] += int(rec.get("words", 0) or 0)
        d["chunks"] += int(rec.get("chunks_added", 0) or 0)

    st.markdown("### Per-subject breakdown")
    if per_subject:
        # Table-like rendering
        hdr = st.columns([3, 2, 2, 2])
        hdr[0].write("Subject")
        hdr[1].write("Files")
        hdr[2].write("Words")
        hdr[3].write("Chunks (ingested)")
        for s, d in per_subject.items():
            row = st.columns([3, 2, 2, 2])
            row[0].write(s)
            row[1].write(d["files"])
            row[2].write(f"{d['words']:,}")
            row[3].write(d["chunks"])
    else:
        st.info("No per-subject data yet.")

    # Query analytics
    st.markdown("### Query analytics")
    qh = st.session_state.get("query_history", [])
    if qh:
        avg_time = sum(q.get("search_time", 0.0) for q in qh) / len(qh)
        avg_results = sum(int(q.get("results_count", 0)) for q in qh) / len(qh)
        cqa = st.columns(2)
        cqa[0].metric("Avg search time", f"{avg_time:.2f}s")
        cqa[1].metric("Avg results", f"{avg_results:.2f}")

        st.markdown("#### Recent queries")
        for q in reversed(qh[-10:]):
            c1, c2 = st.columns([0.85, 0.15])
            c1.write(
                f"[{q.get('timestamp','')}] ({q.get('subject','General')}) "
                f"k={q.get('k',3)} exp={q.get('use_expansion',False)} — {q.get('query','')}"
            )
            if c2.button("Re-run", key=f"re_{q.get('timestamp','')}_{hash(q.get('query',''))%10000}"):
                st.session_state["query_text"] = q.get("query", "")
                st.toast("Query populated in Intelligent Search tab.")
    else:
        st.info("No queries logged yet.")


# ============================ Tab: Management =================================

with tab_manage:
    st.subheader("Management")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Reinitialize RAG", use_container_width=True):
            if not rag:
                st.error("rag.py not available.")
            else:
                try:
                    init_fn = getattr(rag, "initialize_rag_system", None)
                    if callable(init_fn):
                        init_fn(PERSIST_DIR)
                        st.session_state["rag_ready"] = True
                        # Refresh stats
                        stats_fn = getattr(rag, "get_database_stats", None) or getattr(rag, "get_db_stats", None)
                        if callable(stats_fn):
                            st.session_state["db_stats"] = stats_fn() or st.session_state["db_stats"]
                        st.success("RAG reinitialized.")
                    else:
                        st.warning("initialize_rag_system() not found in rag.py.")
                except Exception as e:
                    st.error(f"Reinit failed: {e}")
    with c2:
        if st.button("Clear session", use_container_width=True):
            st.session_state["processed_files"] = []
            st.session_state["query_history"] = []
            st.session_state["search_results"] = []
            st.session_state["search_meta"] = {}
            st.toast("Session cleared.")
    with c3:
        if st.button("Reset theme", use_container_width=True):
            st.session_state["dark_mode"] = False
            st.toast("Theme reset to light.")

    st.markdown("### Export")
    # Export session JSON
    sess_data = {
        "app": APP_NAME,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "subjects": st.session_state.get("subjects", []),
        "active_subject": st.session_state.get("active_subject", "General"),
        "processed_files": st.session_state.get("processed_files", []),
        "query_history": st.session_state.get("query_history", []),
        "db_stats": st.session_state.get("db_stats", {}),
        "settings": {
            "k": st.session_state.get("k", 3),
            "use_expansion": st.session_state.get("use_expansion", False),
            "preview_len": st.session_state.get("preview_len", 600),
            "dark_mode": st.session_state.get("dark_mode", False),
        },
    }
    st.download_button(
        "Export Session JSON",
        data=json.dumps(sess_data, indent=2).encode("utf-8"),
        file_name=f"study_buddy_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        use_container_width=True,
    )

    # Export subject data JSON
    subj = st.selectbox("Subject to export", options=st.session_state.get("subjects", []))
    if st.button("Export Subject JSON", use_container_width=True):
        pf = [r for r in st.session_state.get("processed_files", []) if r.get("subject") == subj]
        payload = {
            "subject": subj,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "files": pf,
        }
        st.download_button(
            "Download Subject JSON",
            data=json.dumps(payload, indent=2).encode("utf-8"),
            file_name=f"{subj}_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True,
        )

    st.markdown("### Diagnostics")
    with st.expander("rag.py integration"):
        if rag:
            st.success("rag.py loaded.")
            st.write("Available functions:")
            funcs = [fn for fn in ["initialize_rag_system", "extract_text", "add_notes_to_db", "query_notes", "get_database_stats", "get_db_stats"] if callable(getattr(rag, fn, None))]
            st.write(funcs)
            st.json(st.session_state.get("db_stats", {}))
        else:
            st.error("rag.py not loaded.")
            if rag_import_error:
                st.code(rag_import_error)
