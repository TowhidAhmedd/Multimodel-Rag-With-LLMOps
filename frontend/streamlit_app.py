







"""
FIXED frontend/streamlit_app.py
Replace your current one with this - fixes the upload error
"""

import os
import time
import requests
# pyrefly: ignore [missing-import]
import streamlit as st

# FIX 1: Use environment variable, fallback to localhost
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "200"))

SUPPORTED_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
]

ACCEPTED_TYPES = ["pdf", "docx", "txt", "mp3", "wav", "m4a", "mp4", "avi", "mov"]

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Multimodal RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  .main-header  { font-size:2.2rem; font-weight:700; color:#1a1a2e; margin-bottom:.2rem; }
  .sub-header   { font-size:1rem;   color:#555;       margin-bottom:1.5rem; }
  .chunk-card   {
    background:#fafafa; border:1px solid #e0e0e0; border-radius:8px;
    padding:12px; margin-bottom:8px; font-size:.88rem; line-height:1.5;
  }
  .source-badge {
    background:#e8f4f8; border-radius:12px; padding:3px 10px;
    font-size:.82rem; color:#1565c0; display:inline-block; margin:2px;
  }
  .conf-high { color:#2e7d32; font-weight:600; }
  .conf-mid  { color:#e65100; font-weight:600; }
  .conf-low  { color:#c62828; font-weight:600; }
  .trace-id  { font-family:monospace; font-size:.78rem; color:#888; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown('<div class="main-header">🧠 Multimodal RAG</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Upload documents, audio, or video — '
    'ask questions grounded strictly in your content.</div>',
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Settings")
    selected_model = st.selectbox("LLM Model", SUPPORTED_MODELS, index=0)
    top_k = st.slider("Retrieved Chunks (Top-K)", min_value=1, max_value=15, value=5)

    st.divider()
    st.header("🔗 System Status")
    try:
        health = requests.get(f"{BACKEND_URL}/health", timeout=5).json()
        status_icon = "🟢" if health.get("status") == "ok" else "🟡"
        st.markdown(f"{status_icon} Backend **{health.get('status', '?')}**")
        st.caption(f"DB: {health.get('database', '?')}")
        st.caption(f"Embed model: {health.get('embedding_model', '?')}")
    except requests.exceptions.ConnectionError:
        st.error("⛔ Backend unreachable")
        st.caption(f"Trying: {BACKEND_URL}")
    except Exception as exc:
        st.warning(f"Status check failed: {exc}")

    st.divider()
    st.header("📊 Cache")
    try:
        metrics_data = requests.get(f"{BACKEND_URL}/metrics", timeout=5).json()
        col_a, col_b = st.columns(2)
        col_a.metric("Entries", metrics_data.get("cache_size", 0))
        col_b.metric("Max", metrics_data.get("cache_maxsize", 0))
        if st.button("🗑️ Clear Cache"):
            r = requests.delete(f"{BACKEND_URL}/cache", timeout=5)
            if r.status_code == 200:
                st.success(f"Cleared {r.json().get('cleared', 0)} entries")
            else:
                st.error("Clear failed")
    except Exception:
        st.caption("Metrics unavailable")

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_upload, tab_query = st.tabs(["📁 Upload Files", "💬 Ask Questions"])

# ────────────────────────────────────────────────────────────────────────────
# TAB 1 – Upload
# ────────────────────────────────────────────────────────────────────────────
with tab_upload:
    st.subheader("Upload Documents, Audio, or Video")
    st.caption(
        f"Supported: {', '.join('.' + t for t in ACCEPTED_TYPES)} "
        f"· Max {MAX_UPLOAD_MB} MB per file"
    )

    uploaded_files = st.file_uploader(
        "Choose one or more files",
        accept_multiple_files=True,
        type=ACCEPTED_TYPES,
        help="All formats are ingested into the same vector store. "
             "You can filter queries by file name.",
    )

    if uploaded_files:
        if st.button("🚀 Ingest Selected Files", type="primary"):
            for uf in uploaded_files:
                st.markdown(f"**Processing: {uf.name}**")
                progress_bar = st.progress(0, text="Uploading file…")
                status_area = st.empty()

                # ── Step 1: Post to /upload-async (returns immediately) ───────
                try:
                    resp = requests.post(
                        f"{BACKEND_URL}/upload-async",
                        files={"file": (uf.name, uf.getvalue(), uf.type or "application/octet-stream")},
                        timeout=60,   # only waiting for file-save, not ingestion
                    )
                except requests.exceptions.Timeout:
                    st.error(f"❌ **{uf.name}**: Timed out while uploading. File may be too large.")
                    continue
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to backend. Check BACKEND_URL in environment.")
                    continue
                except Exception as exc:
                    st.error(f"❌ **{uf.name}**: {type(exc).__name__}: {exc}")
                    continue

                if resp.status_code != 200:
                    try:
                        detail = resp.json().get("detail", resp.text)
                    except Exception:
                        detail = resp.text[:300]
                    st.error(f"❌ **{uf.name}** (HTTP {resp.status_code}): {detail}")
                    continue

                try:
                    job_data = resp.json()
                    job_id = job_data["job_id"]
                except Exception as exc:
                    raw = resp.text[:300]
                    st.error(f"❌ **{uf.name}**: Bad response from backend — {exc} | raw: {raw}")
                    continue

                progress_bar.progress(20, text="File uploaded — ingestion running in background…")
                status_area.info(f"🔄 Job `{job_id}` queued. Polling for result…")

                # ── Step 2: Poll GET /jobs/{job_id} until done/error ──────────
                MAX_POLL_SECONDS = 600   # 10 minutes max
                POLL_INTERVAL   = 3     # seconds between checks
                elapsed = 0
                dots = 0

                while elapsed < MAX_POLL_SECONDS:
                    time.sleep(POLL_INTERVAL)
                    elapsed += POLL_INTERVAL
                    dots = (dots + 1) % 4
                    dot_str = "." * (dots + 1)

                    try:
                        poll = requests.get(
                            f"{BACKEND_URL}/jobs/{job_id}",
                            timeout=10,
                        )
                        poll_data = poll.json()
                    except Exception as exc:
                        status_area.warning(f"⚠️ Poll error (will retry): {exc}")
                        continue

                    job_status = poll_data.get("status", "unknown")
                    pct = min(20 + int(elapsed / MAX_POLL_SECONDS * 75), 95)
                    progress_bar.progress(pct, text=f"Ingesting{dot_str} ({elapsed}s elapsed)")

                    if job_status == "done":
                        progress_bar.progress(100, text="✅ Complete!")
                        icon = {"text": "📄", "audio": "🎧", "video": "🎬"}.get(
                            poll_data.get("modality", ""), "📄"
                        )
                        status_area.empty()
                        st.success(
                            f"{icon} **{poll_data.get('source_file', uf.name)}** ingested — "
                            f"{poll_data.get('chunk_count', 0)} chunks "
                            f"({poll_data.get('modality', '?')})"
                        )
                        break

                    elif job_status == "error":
                        progress_bar.progress(100, text="❌ Failed")
                        status_area.empty()
                        st.error(f"❌ **{uf.name}** ingestion failed: {poll_data.get('error', 'unknown error')}")
                        break

                    else:
                        status_area.info(f"🔄 Status: `{job_status}` — {elapsed}s elapsed…")
                else:
                    st.warning(
                        f"⚠️ **{uf.name}**: Ingestion is still running after {MAX_POLL_SECONDS}s. "
                        f"Check backend logs for job `{job_id}`."
                    )

# ────────────────────────────────────────────────────────────────────────────
# TAB 2 – Query
# ────────────────────────────────────────────────────────────────────────────
with tab_query:
    st.subheader("Ask a Question")

    source_filter = st.text_input(
        "Filter by source file (optional)",
        placeholder="e.g. quarterly_report.pdf  —  leave blank to search all files",
        help="Enter the exact filename (as shown after upload) to restrict the search.",
    )

    query = st.text_area(
        "Your question",
        placeholder=(
            "What are the main findings?\n"
            "Summarize this meeting recording.\n"
            "What did the speaker say about revenue?"
        ),
        height=130,
    )

    col_ask, col_clear = st.columns([1, 6])
    ask_clicked = col_ask.button("🔍 Ask", type="primary", disabled=not query.strip())
    if col_clear.button("↺ Clear"):
        st.rerun()

    if ask_clicked and query.strip():
        payload: dict = {
            "query": query.strip(),
            "model": selected_model,
            "top_k": top_k,
        }
        if source_filter.strip():
            payload["source_file"] = source_filter.strip()

        with st.spinner("Retrieving context and generating answer …"):
            try:
                resp = requests.post(
                    f"{BACKEND_URL}/query",
                    json=payload,
                    timeout=120,
                )
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to backend.")
                st.stop()
            except Exception as exc:
                st.error(f"Request failed: {exc}")
                st.stop()

        if resp.status_code != 200:
            try:
                detail = resp.json().get('detail', resp.text)
            except:
                detail = resp.text[:200]
            st.error(f"Backend error {resp.status_code}: {detail}")
            st.stop()

        try:
            data = resp.json()
        except ValueError:
            st.error("Backend returned invalid JSON. Check backend logs.")
            st.stop()

        m = data.get("metrics", {})
        confidence = m.get("confidence_score", 0.5)

        # ── Answer ────────────────────────────────────────────────────────────
        st.divider()
        st.subheader("📝 Answer")

        if data.get("from_cache"):
            st.info("⚡ Served from cache", icon="💾")

        st.markdown(data["answer"])

        # ── Key metrics row ───────────────────────────────────────────────────
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("⏱️ Latency",  f"{data['latency_ms']:.0f} ms")
        c2.metric("📦 Chunks",   m.get("retrieval_count", 0))
        c3.metric("📏 Context",  f"{m.get('context_length_chars', 0):,} ch")
        c4.metric("📝 Answer",   f"{m.get('response_length_chars', 0):,} ch")
        c5.metric("🤖 Model",    data["model"].split("-")[0])

        conf_class = "conf-high" if confidence >= 0.7 else "conf-mid" if confidence >= 0.4 else "conf-low"
        st.markdown(
            f'<p class="{conf_class}">🎯 Confidence: {confidence:.0%}</p>',
            unsafe_allow_html=True,
        )

        if data.get("trace_id"):
            st.markdown(
                f'<p class="trace-id">🔭 LangSmith Trace: {data["trace_id"]}</p>',
                unsafe_allow_html=True,
            )
        st.caption(f"Request ID: {data.get('request_id', 'N/A')}")

        # ── Source references ─────────────────────────────────────────────────
        if data.get("sources"):
            st.subheader("📎 Source References")
            for src in data["sources"]:
                mod = src.get("modality", "text")
                icon = {"text": "📄", "audio": "🎧", "video": "🎬"}.get(mod, "📄")
                label = src["source"]
                if src.get("page"):
                    label += f" · Page {src['page']}"
                if src.get("start_timestamp") is not None:
                    label += f" · {src['start_timestamp']:.1f}s – {src.get('end_timestamp', 0):.1f}s"
                st.markdown(
                    f'<span class="source-badge">{icon} {label}</span>',
                    unsafe_allow_html=True,
                )

        # ── Retrieved chunks ──────────────────────────────────────────────────
        if data.get("chunks"):
            with st.expander(f"🔎 Retrieved Chunks ({len(data['chunks'])})", expanded=False):
                for i, chunk in enumerate(data["chunks"]):
                    mod = chunk.get("modality", "text")
                    icon = {"text": "📄", "audio": "🎧", "video": "🎬"}.get(mod, "📄")

                    header_parts = [f"{icon} Chunk {i + 1}", chunk["source_file"]]
                    if chunk.get("page_number"):
                        header_parts.append(f"Page {chunk['page_number']}")
                    if chunk.get("start_timestamp") is not None:
                        header_parts.append(f"{chunk['start_timestamp']:.1f}s")

                    score_parts = []
                    if chunk.get("rerank_score") is not None:
                        score_parts.append(f"Re-rank: {chunk['rerank_score']:.3f}")
                    elif chunk.get("score") is not None:
                        score_parts.append(f"Score: {chunk['score']:.3f}")

                    header = " · ".join(header_parts)
                    score_str = f" <em>({', '.join(score_parts)})</em>" if score_parts else ""
                    content = chunk["content"].replace("<", "&lt;").replace(">", "&gt;")

                    st.markdown(
                        f'<div class="chunk-card">'
                        f'<strong>{header}</strong>{score_str}<br><br>{content}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        # ── Evaluation metrics ────────────────────────────────────────────────
        if m:
            with st.expander("📊 Evaluation Metrics", expanded=False):
                col_l, col_r = st.columns(2)
                with col_l:
                    st.json({
                        "confidence_score":       m.get("confidence_score"),
                        "avg_retrieval_score":    m.get("avg_retrieval_score"),
                        "avg_rerank_score":       m.get("avg_rerank_score"),
                        "has_sufficient_evidence": m.get("has_sufficient_evidence"),
                    })
                with col_r:
                    st.json({
                        "retrieval_count":       m.get("retrieval_count"),
                        "context_length_chars":  m.get("context_length_chars"),
                        "response_length_chars": m.get("response_length_chars"),
                        "latency_ms":            m.get("latency_ms"),
                    })







# """
# frontend/streamlit_app.py
# Multimodal RAG – Streamlit User Interface
# """

# import os
# import requests
# # pyrefly: ignore [missing-import]
# import streamlit as st

# BACKEND_URL = os.getenv("BACKEND_URL", "https://multimodel-rag-with-llmops-backend.onrender.com").rstrip("/")
# MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "200"))

# SUPPORTED_MODELS = [
#     "llama-3.3-70b-versatile",
#     "llama-3.1-8b-instant",
#     "mixtral-8x7b-32768",
# ]

# ACCEPTED_TYPES = ["pdf", "docx", "txt", "mp3", "wav", "m4a", "mp4", "avi", "mov"]

# # ── Page config ───────────────────────────────────────────────────────────────

# st.set_page_config(
#     page_title="Multimodal RAG",
#     page_icon="🧠",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

# # ── CSS ───────────────────────────────────────────────────────────────────────

# st.markdown("""
# <style>
#   .main-header  { font-size:2.2rem; font-weight:700; color:#1a1a2e; margin-bottom:.2rem; }
#   .sub-header   { font-size:1rem;   color:#555;       margin-bottom:1.5rem; }
#   .chunk-card   {
#     background:#fafafa; border:1px solid #e0e0e0; border-radius:8px;
#     padding:12px; margin-bottom:8px; font-size:.88rem; line-height:1.5;
#   }
#   .source-badge {
#     background:#e8f4f8; border-radius:12px; padding:3px 10px;
#     font-size:.82rem; color:#1565c0; display:inline-block; margin:2px;
#   }
#   .conf-high { color:#2e7d32; font-weight:600; }
#   .conf-mid  { color:#e65100; font-weight:600; }
#   .conf-low  { color:#c62828; font-weight:600; }
#   .trace-id  { font-family:monospace; font-size:.78rem; color:#888; }
# </style>
# """, unsafe_allow_html=True)

# # ── Header ────────────────────────────────────────────────────────────────────

# st.markdown('<div class="main-header">🧠 Multimodal RAG</div>', unsafe_allow_html=True)
# st.markdown(
#     '<div class="sub-header">Upload documents, audio, or video — '
#     'ask questions grounded strictly in your content.</div>',
#     unsafe_allow_html=True,
# )

# # ── Sidebar ───────────────────────────────────────────────────────────────────

# with st.sidebar:
#     st.header("⚙️ Settings")
#     selected_model = st.selectbox("LLM Model", SUPPORTED_MODELS, index=0)
#     top_k = st.slider("Retrieved Chunks (Top-K)", min_value=1, max_value=15, value=5)

#     st.divider()
#     st.header("🔗 System Status")
#     try:
#         health = requests.get(f"{BACKEND_URL}/health", timeout=200).json()
#         status_icon = "🟢" if health.get("status") == "ok" else "🟡"
#         st.markdown(f"{status_icon} Backend **{health.get('status', '?')}**")
#         st.caption(f"DB: {health.get('database', '?')}")
#         st.caption(f"Embed model: {health.get('embedding_model', '?')}")
#     except requests.exceptions.ConnectionError:
#         st.error("⛔ Backend unreachable")
#     except Exception as exc:
#         st.warning(f"Status check failed: {exc}")



#     st.divider()
#     st.header("📊 Cache")
#     try:
#         metrics_data = requests.get(f"{BACKEND_URL}/metrics", timeout=200).json()
#         col_a, col_b = st.columns(2)
#         col_a.metric("Entries", metrics_data.get("cache_size", 0))
#         col_b.metric("Max", metrics_data.get("cache_maxsize", 0))
#         if st.button("🗑️ Clear Cache"):
#             r = requests.delete(f"{BACKEND_URL}/cache", timeout=200)
#             if r.status_code == 200:
#                 st.success(f"Cleared {r.json().get('cleared', 0)} entries")
#             else:
#                 st.error("Clear failed")
#     except Exception:
#         st.caption("Metrics unavailable")

# # ── Tabs ──────────────────────────────────────────────────────────────────────

# tab_upload, tab_query = st.tabs(["📁 Upload Files", "💬 Ask Questions"])

# # ────────────────────────────────────────────────────────────────────────────
# # TAB 1 – Upload
# # ────────────────────────────────────────────────────────────────────────────
# with tab_upload:
#     st.subheader("Upload Documents, Audio, or Video")
#     st.caption(
#         f"Supported: {', '.join('.' + t for t in ACCEPTED_TYPES)} "
#         f"· Max {MAX_UPLOAD_MB} MB per file"
#     )

#     uploaded_files = st.file_uploader(
#         "Choose one or more files",
#         accept_multiple_files=True,
#         type=ACCEPTED_TYPES,
#         help="All formats are ingested into the same vector store. "
#              "You can filter queries by file name.",
#     )

#     if uploaded_files:
#         if st.button("🚀 Ingest Selected Files", type="primary"):
#             for uf in uploaded_files:
#                 with st.spinner(f"Ingesting **{uf.name}** …"):
#                     try:
#                         resp = requests.post(
#                             f"{BACKEND_URL}/upload",
#                             files={"file": (uf.name, uf.getvalue(), uf.type or "application/octet-stream")},
#                             timeout=600,   # allow up to 10 min for large video files
#                         )
#                         if resp.status_code == 200:
#                             d = resp.json()
#                             icon = {"text": "📄", "audio": "🎧", "video": "🎬"}.get(d.get("modality", ""), "📄")
#                             st.success(
#                                 f"{icon} **{d['source_file']}** ingested — "
#                                 f"{d['chunk_count']} chunks ({d['modality']})"
#                             )
#                         else:
#                             detail = resp.json().get("detail", resp.text)
#                             st.error(f"❌ **{uf.name}**: {detail}")
#                     except requests.exceptions.ConnectionError:
#                         st.error("Cannot connect to backend. Is it running?")
#                     except Exception as exc:
#                         st.error(f"Unexpected error uploading **{uf.name}**: {exc}")

# # ────────────────────────────────────────────────────────────────────────────
# # TAB 2 – Query
# # ────────────────────────────────────────────────────────────────────────────
# with tab_query:
#     st.subheader("Ask a Question")

#     source_filter = st.text_input(
#         "Filter by source file (optional)",
#         placeholder="e.g. quarterly_report.pdf  —  leave blank to search all files",
#         help="Enter the exact filename (as shown after upload) to restrict the search.",
#     )

#     query = st.text_area(
#         "Your question",
#         placeholder=(
#             "What are the main findings?\n"
#             "Summarize this meeting recording.\n"
#             "What did the speaker say about revenue?"
#         ),
#         height=130,
#     )

#     col_ask, col_clear = st.columns([1, 6])
#     ask_clicked = col_ask.button("🔍 Ask", type="primary", disabled=not query.strip())
#     if col_clear.button("↺ Clear"):
#         st.rerun()

#     if ask_clicked and query.strip():
#         payload: dict = {
#             "query": query.strip(),
#             "model": selected_model,
#             "top_k": top_k,
#         }
#         if source_filter.strip():
#             payload["source_file"] = source_filter.strip()

#         with st.spinner("Retrieving context and generating answer …"):
#             try:
#                 resp = requests.post(
#                     f"{BACKEND_URL}/query",
#                     json=payload,
#                     timeout=120,
#                 )
#             except requests.exceptions.ConnectionError:
#                 st.error("Cannot connect to backend.")
#                 st.stop()
#             except Exception as exc:
#                 st.error(f"Request failed: {exc}")
#                 st.stop()

#         if resp.status_code != 200:
#             st.error(f"Backend error {resp.status_code}: {resp.json().get('detail', resp.text)}")
#             st.stop()

#         data = resp.json()
#         m = data.get("metrics", {})
#         confidence = m.get("confidence_score", 0.5)

#         # ── Answer ────────────────────────────────────────────────────────────
#         st.divider()
#         st.subheader("📝 Answer")

#         if data.get("from_cache"):
#             st.info("⚡ Served from cache", icon="💾")

#         st.markdown(data["answer"])

#         # ── Key metrics row ───────────────────────────────────────────────────
#         c1, c2, c3, c4, c5 = st.columns(5)
#         c1.metric("⏱️ Latency",  f"{data['latency_ms']:.0f} ms")
#         c2.metric("📦 Chunks",   m.get("retrieval_count", 0))
#         c3.metric("📏 Context",  f"{m.get('context_length_chars', 0):,} ch")
#         c4.metric("📝 Answer",   f"{m.get('response_length_chars', 0):,} ch")
#         c5.metric("🤖 Model",    data["model"].split("-")[0])

#         conf_class = "conf-high" if confidence >= 0.7 else "conf-mid" if confidence >= 0.4 else "conf-low"
#         st.markdown(
#             f'<p class="{conf_class}">🎯 Confidence: {confidence:.0%}</p>',
#             unsafe_allow_html=True,
#         )

#         if data.get("trace_id"):
#             st.markdown(
#                 f'<p class="trace-id">🔭 LangSmith Trace: {data["trace_id"]}</p>',
#                 unsafe_allow_html=True,
#             )
#         st.caption(f"Request ID: {data.get('request_id', 'N/A')}")

#         # ── Source references ─────────────────────────────────────────────────
#         if data.get("sources"):
#             st.subheader("📎 Source References")
#             for src in data["sources"]:
#                 mod = src.get("modality", "text")
#                 icon = {"text": "📄", "audio": "🎧", "video": "🎬"}.get(mod, "📄")
#                 label = src["source"]
#                 if src.get("page"):
#                     label += f" · Page {src['page']}"
#                 if src.get("start_timestamp") is not None:
#                     label += f" · {src['start_timestamp']:.1f}s – {src.get('end_timestamp', 0):.1f}s"
#                 st.markdown(
#                     f'<span class="source-badge">{icon} {label}</span>',
#                     unsafe_allow_html=True,
#                 )

#         # ── Retrieved chunks ──────────────────────────────────────────────────
#         if data.get("chunks"):
#             with st.expander(f"🔎 Retrieved Chunks ({len(data['chunks'])})", expanded=False):
#                 for i, chunk in enumerate(data["chunks"]):
#                     mod = chunk.get("modality", "text")
#                     icon = {"text": "📄", "audio": "🎧", "video": "🎬"}.get(mod, "📄")

#                     header_parts = [f"{icon} Chunk {i + 1}", chunk["source_file"]]
#                     if chunk.get("page_number"):
#                         header_parts.append(f"Page {chunk['page_number']}")
#                     if chunk.get("start_timestamp") is not None:
#                         header_parts.append(f"{chunk['start_timestamp']:.1f}s")

#                     score_parts = []
#                     if chunk.get("rerank_score") is not None:
#                         score_parts.append(f"Re-rank: {chunk['rerank_score']:.3f}")
#                     elif chunk.get("score") is not None:
#                         score_parts.append(f"Score: {chunk['score']:.3f}")

#                     header = " · ".join(header_parts)
#                     score_str = f" <em>({', '.join(score_parts)})</em>" if score_parts else ""
#                     content = chunk["content"].replace("<", "&lt;").replace(">", "&gt;")

#                     st.markdown(
#                         f'<div class="chunk-card">'
#                         f'<strong>{header}</strong>{score_str}<br><br>{content}'
#                         f'</div>',
#                         unsafe_allow_html=True,
#                     )

#         # ── Evaluation metrics ────────────────────────────────────────────────
#         if m:
#             with st.expander("📊 Evaluation Metrics", expanded=False):
#                 col_l, col_r = st.columns(2)
#                 with col_l:
#                     st.json({
#                         "confidence_score":       m.get("confidence_score"),
#                         "avg_retrieval_score":    m.get("avg_retrieval_score"),
#                         "avg_rerank_score":       m.get("avg_rerank_score"),
#                         "has_sufficient_evidence": m.get("has_sufficient_evidence"),
#                     })
#                 with col_r:
#                     st.json({
#                         "retrieval_count":       m.get("retrieval_count"),
#                         "context_length_chars":  m.get("context_length_chars"),
#                         "response_length_chars": m.get("response_length_chars"),
#                         "latency_ms":            m.get("latency_ms"),
#                     })
