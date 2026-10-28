import streamlit as st
import pandas as pd
import numpy as np
import io

from processor import (
    extract_pdf_pages,
    extract_paper_dimensions,
    extract_key_concepts,
    split_sentences_with_provenance,
    summarize_text
)
from search_engine import MultiPaperSearchEngine
from knowledge_graph import (
    build_multipaper_knowledge_graph,
    render_plotly_graph,
    render_matplotlib_graph,
    get_graph_metrics,
    NODE_COLORS
)
from gap_analyzer import discover_research_gaps, analyze_cross_paper_claims


# --------------------------------------------------
# 1. PAGE CONFIGURATION & STYLING
# --------------------------------------------------

st.set_page_config(
    page_title="LitSense — Research Intelligence",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern academic UI
st.markdown("""
<style>
/* Main App Styling */
.stApp {
    background-color: #f8fafc;
    color: #0f172a;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}

/* Adjust Block Container Spacing */
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}

/* Header & Typography */
h1, h2, h3, h4 {
    color: #0f172a !important;
    font-weight: 700 !important;
    letter-spacing: -0.025em !important;
}

/* Hero Banner */
.litsense-hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #312e81 100%);
    padding: 30px 36px;
    border-radius: 18px;
    color: #ffffff !important;
    margin-bottom: 24px;
    box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.2);
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.litsense-hero h1 {
    color: #ffffff !important;
    font-size: 32px !important;
    margin-bottom: 6px !important;
    font-weight: 800 !important;
}

.litsense-hero p {
    color: #cbd5e1 !important;
    font-size: 15px !important;
    margin-bottom: 0 !important;
    max-width: 900px;
}

/* Metric Cards */
div[data-testid="stMetric"] {
    background: #ffffff;
    padding: 16px 20px;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
}

div[data-testid="stMetricLabel"] {
    color: #64748b !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

div[data-testid="stMetricValue"] {
    color: #3730a3 !important;
    font-size: 26px !important;
    font-weight: 800 !important;
}

/* Academic Paper Cards */
.paper-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 18px;
    box-shadow: 0 1px 4px rgba(15, 23, 42, 0.04);
}

.badge-tag {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 9999px;
    font-size: 12px;
    font-weight: 600;
    margin-right: 6px;
}
.badge-green { background-color: #dcfce7; color: #166534; }
.badge-red { background-color: #fee2e2; color: #991b1b; }
.badge-blue { background-color: #dbeafe; color: #1e40af; }
.badge-amber { background-color: #fef3c7; color: #92400e; }

/* Custom Buttons */
.stButton > button {
    border-radius: 10px;
    font-weight: 600;
    padding: 0.5rem 1.1rem;
    transition: all 0.2s ease;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background-color: #f1f5f9;
    border-right: 1px solid #e2e8f0;
}

/* Clean Alert Boxes */
div[data-testid="stAlert"] {
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 2. DEMO RESEARCH PAPERS GENERATOR
# --------------------------------------------------

def generate_demo_papers() -> list:
    """Generate 3 realistic demo academic PDFs for instant demonstration."""
    import pymupdf

    demo_specs = [
        {
            "filename": "Vaswani_Attention_Transformers_2017.pdf",
            "pages": [
                (
                    "Vaswani et al. (2017) - Attention Is All You Need\n\n"
                    "1. Introduction and Problem Formulation\n"
                    "In this paper, we address the problem of sequential computation constraints in recurrent neural networks. "
                    "Traditional approaches suffer from sequential training bottlenecks and lack of parallelization. "
                    "The main challenge is capturing long-range dependencies without quadratic recurrence."
                ),
                (
                    "2. Proposed Architecture\n"
                    "We propose the Transformer, an architecture based entirely on multi-head self-attention mechanisms. "
                    "We evaluate on the WMT 2014 English-to-German benchmark and WMT 2014 English-to-French dataset. "
                    "The model consists of stacked encoder-decoder layers with sinusoidal positional encodings."
                ),
                (
                    "3. Results, Limitations, and Future Work\n"
                    "Experimental results show that the Transformer achieves state-of-the-art with 28.4 BLEU score, consistently outperforming recurrent baselines. "
                    "However, a critical limitation is that self-attention scales quadratically with input sequence length, incurring heavy memory overhead. "
                    "In the future, we plan to extend attention to handle ultra-long sequences using sparse local receptive fields."
                )
            ]
        },
        {
            "filename": "Dao_FlashAttention_Fast_Memory_2022.pdf",
            "pages": [
                (
                    "Dao et al. (2022) - FlashAttention: Fast and Memory-Efficient Exact Attention\n\n"
                    "1. Problem Formulation and Motivation\n"
                    "The objective of this research is resolving memory-bandwidth bottlenecks in standard self-attention. "
                    "Existing methods suffer from excessive read/write operations between High Bandwidth Memory and SRAM on GPUs. "
                    "The limitation of existing attention implementations is memory latency rather than raw compute speed."
                ),
                (
                    "2. Methodology and Algorithms\n"
                    "We introduce FlashAttention, an IO-aware exact attention algorithm using GPU tiling and kernel fusion. "
                    "We evaluate on ImageNet classification benchmarks and the Long Range Arena sequence suite. "
                    "The proposed framework computes softmax dynamically without materializing the full intermediate N-by-N attention matrix."
                ),
                (
                    "3. Findings and Limitations\n"
                    "Results demonstrate 2x to 4x wall-clock speedups over standard PyTorch attention with reduced memory footprint. "
                    "However, our approach fails when deployed on non-CUDA hardware architectures or legacy CPUs due to custom SRAM kernel requirements. "
                    "Future work involves adapting FlashAttention for lower bit-width FP8 inference and distributed multi-GPU sharding."
                )
            ]
        },
        {
            "filename": "Beltagy_Longformer_Long_Documents_2020.pdf",
            "pages": [
                (
                    "Beltagy et al. (2020) - Longformer: The Long-Document Transformer\n\n"
                    "1. Motivation and Problem\n"
                    "Standard transformer architectures fail to scale to long documents such as scientific papers and books. "
                    "The main challenge is that full self-attention requires 512 token limits, causing truncation of vital contextual evidence. "
                    "Existing methods suffer from severe context loss when evaluating long-form QA."
                ),
                (
                    "2. Method and Experimental Setup\n"
                    "We develop Longformer, a modified transformer combining sliding window local attention with sparse global attention tokens. "
                    "We evaluate on the WikiHop multi-hop reasoning dataset and Hyperpartisan news corpus. "
                    "The proposed framework scales linearly with sequence length O(N)."
                ),
                (
                    "3. Findings and Open Directions\n"
                    "Experimental results show state-of-the-art results on WikiHop, consistently outperforming RoBERTa baselines. "
                    "A notable limitation is that configuring global attention tokens requires task-specific manual heuristics. "
                    "Further research is needed to learn optimal global token placement automatically during pretraining."
                )
            ]
        }
    ]

    processed_demo = []
    for spec in demo_specs:
        doc = pymupdf.open()
        for page_text in spec["pages"]:
            p = doc.new_page()
            p.insert_textbox(pymupdf.Rect(50, 50, 550, 750), page_text)
        pdf_bytes = doc.tobytes()
        doc.close()

        extracted = extract_pdf_pages(pdf_bytes, filename=spec["filename"])
        dims = extract_paper_dimensions(extracted["pages"], spec["filename"])
        extracted["dimensions"] = dims
        extracted["key_concepts"] = extract_key_concepts(extracted["full_text"], max_concepts=10)
        processed_demo.append(extracted)

    return processed_demo


# --------------------------------------------------
# 3. SESSION STATE INITIALIZATION
# --------------------------------------------------

if "papers" not in st.session_state:
    st.session_state["papers"] = []

if "search_engine" not in st.session_state:
    st.session_state["search_engine"] = MultiPaperSearchEngine()

if "knowledge_graph" not in st.session_state:
    st.session_state["knowledge_graph"] = None


# --------------------------------------------------
# 4. SIDEBAR NAVIGATION
# --------------------------------------------------

with st.sidebar:
    st.markdown("### 📚 **LitSense V1**")
    st.caption("Intelligent Literature Review & Research Gap Discovery")
    st.divider()

    navigation = st.radio(
        "Navigation",
        [
            "🏢 Workspace & Papers",
            "📊 Paper Comparison Matrix",
            "🔍 Research Gap Explorer",
            "🕸️ Evidence Knowledge Graph",
            "💬 Ask the Literature",
            "⚖️ Claim Alignment (Agreement & Divergence)",
            "📚 NLP Methodology"
        ],
        index=0
    )

    st.divider()

    # Quick Demo Loader
    st.markdown("#### ⚡ Viva Quick Demo")
    if st.button("Load 3 Demo Academic Papers", use_container_width=True, type="secondary"):
        with st.spinner("Generating and analyzing 3 sample research papers..."):
            demo_papers = generate_demo_papers()
            st.session_state["papers"] = demo_papers
            # Reindex search engine
            engine = MultiPaperSearchEngine()
            engine.index_papers(demo_papers)
            st.session_state["search_engine"] = engine
            # Rebuild graph
            st.session_state["knowledge_graph"] = build_multipaper_knowledge_graph(demo_papers)
            st.success("Loaded 3 NLP research papers!")
            st.rerun()

    if st.session_state["papers"]:
        if st.button("Clear All Papers", use_container_width=True):
            st.session_state["papers"] = []
            st.session_state["search_engine"] = MultiPaperSearchEngine()
            st.session_state["knowledge_graph"] = None
            st.rerun()

    st.divider()
    st.caption("LitSense V1 • B.Tech CSE / Data Science Project • Streamlit & Local NLP")


# --------------------------------------------------
# 5. HERO BANNER
# --------------------------------------------------

st.markdown("""
<div class="litsense-hero">
    <h1>LitSense — Literature Review & Research Gap Discovery</h1>
    <p>
        Accelerate systematic academic literature reviews using local natural language processing. 
        Extract multi-paper dimensions, discover evidence-backed research gaps, inspect knowledge graphs, 
        and query the literature with page-level citation provenance.
    </p>
</div>
""", unsafe_allow_html=True)


papers = st.session_state["papers"]
total_papers = len(papers)
total_pages = sum(p.get("total_pages", 0) for p in papers)
total_words = sum(len(p.get("full_text", "").split()) for p in papers)


# ==================================================
# SECTION 1: WORKSPACE & PAPERS
# ==================================================

if navigation == "🏢 Workspace & Papers":
    st.subheader("Multi-Paper Workspace")
    st.write("Upload multiple research papers (PDF) to begin systematic cross-paper extraction and analysis.")

    # Workspace Metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Papers Loaded", total_papers)
    with m2:
        st.metric("Total Pages", total_pages)
    with m3:
        st.metric("Total Words Extracted", f"{total_words:,}")
    with m4:
        gap_count = 0
        if papers:
            gap_data = discover_research_gaps(papers)
            gap_count = len(gap_data.get("candidate_gaps", []))
        st.metric("Identified Candidate Gaps", gap_count)

    st.markdown("---")

    # File Uploader
    uploaded_files = st.file_uploader(
        "Upload Research Papers (PDF)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more academic research papers in PDF format."
    )

    if uploaded_files:
        if st.button("Process Uploaded Papers", type="primary"):
            new_papers = []
            progress_bar = st.progress(0)
            status_text = st.empty()

            for idx, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Extracting: {uploaded_file.name} ({idx+1}/{len(uploaded_files)})...")
                pdf_bytes = uploaded_file.getvalue()
                extracted = extract_pdf_pages(pdf_bytes, filename=uploaded_file.name)

                if extracted["status"] == "Success":
                    dims = extract_paper_dimensions(extracted["pages"], uploaded_file.name)
                    extracted["dimensions"] = dims
                    extracted["key_concepts"] = extract_key_concepts(extracted["full_text"], max_concepts=10)
                else:
                    extracted["dimensions"] = {}
                    extracted["key_concepts"] = []

                new_papers.append(extracted)
                progress_bar.progress((idx + 1) / len(uploaded_files))

            status_text.text("Indexing multi-paper retrieval engine and knowledge graph...")
            engine = MultiPaperSearchEngine()
            engine.index_papers(new_papers)
            st.session_state["search_engine"] = engine
            st.session_state["knowledge_graph"] = build_multipaper_knowledge_graph(new_papers)
            st.session_state["papers"] = new_papers

            progress_bar.empty()
            status_text.empty()
            st.success(f"Successfully processed {len(new_papers)} research papers!")
            st.rerun()

    # Display Current Workspace Table
    if papers:
        st.subheader("Indexed Documents in Workspace")
        table_rows = []
        for p in papers:
            table_rows.append({
                "Document Name": p.get("filename", "Unknown"),
                "Pages": p.get("total_pages", 0),
                "Characters": len(p.get("full_text", "")),
                "Extraction Status": p.get("status", "Unknown"),
                "Extracted Dimensions": len([k for k, v in p.get("dimensions", {}).items() if v.get("evidence")]),
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

        st.markdown("#### Document Content Viewer")
        selected_paper_name = st.selectbox(
            "Select a document to inspect extracted pages",
            [p["filename"] for p in papers]
        )
        selected_paper = next((p for p in papers if p["filename"] == selected_paper_name), None)

        if selected_paper:
            with st.expander(f"View Extracted Text for {selected_paper_name} ({selected_paper['total_pages']} Pages)", expanded=False):
                for page in selected_paper.get("pages", []):
                    st.markdown(f"**--- Page {page['page_number']} ---**")
                    st.text_area(
                        label=f"Page {page['page_number']}",
                        value=page["text"],
                        height=160,
                        disabled=True,
                        key=f"viewer_{selected_paper_name}_{page['page_number']}"
                    )
    else:
        st.info("No research papers uploaded yet. Upload your PDF files above or click **'Load 3 Demo Academic Papers'** in the sidebar for an instant demonstration.")


# ==================================================
# SECTION 2: PAPER COMPARISON MATRIX
# ==================================================

elif navigation == "📊 Paper Comparison Matrix":
    st.subheader("Systematic Paper Comparison Matrix")
    st.write(
        "Structured comparison across core academic dimensions: Problem, Methodology, Datasets, Findings, Limitations, and Future Work. "
        "Every extracted cell links to verbatim source evidence and page numbers."
    )

    if not papers:
        st.info("Please upload papers in the **Workspace** tab or load demo papers from the sidebar to view the comparison matrix.")
    else:
        # Build comparison rows
        matrix_rows = []
        for p in papers:
            dims = p.get("dimensions", {})
            matrix_rows.append({
                "Paper Title": p.get("filename", "Paper"),
                "Research Problem": dims.get("Problem", {}).get("summary", "Not detected"),
                "Methodology": dims.get("Methodology", {}).get("summary", "Not detected"),
                "Datasets / Setup": dims.get("Datasets", {}).get("summary", "Not detected"),
                "Key Findings": dims.get("Findings", {}).get("summary", "Not detected"),
                "Limitations": dims.get("Limitations", {}).get("summary", "Not detected"),
                "Future Work": dims.get("Future Work", {}).get("summary", "Not detected"),
            })

        st.dataframe(pd.DataFrame(matrix_rows), use_container_width=True)

        st.markdown("---")
        st.subheader("Evidence Inspector & Citation Provenance")
        st.caption("Inspect verbatim excerpts, cue match confidences, and exact page provenance for any paper.")

        selected_p = st.selectbox("Select paper for deep evidence inspection", [p["filename"] for p in papers])
        target_paper = next((p for p in papers if p["filename"] == selected_p), None)

        if target_paper:
            dims = target_paper.get("dimensions", {})
            col_a, col_b = st.columns(2)

            categories = list(dims.keys())
            half = len(categories) // 2

            with col_a:
                for cat in categories[:half]:
                    info = dims[cat]
                    with st.expander(f"📌 {cat} — Status: {info.get('status', 'Unknown')}", expanded=True):
                        st.markdown(f"**Summary:** {info.get('summary', 'N/A')}")
                        if info.get("evidence"):
                            st.markdown("**Verbatim Source Evidence:**")
                            for ev in info["evidence"]:
                                st.markdown(f"- *\"{ev['text']}\"* `[Page {ev.get('page_number', 1)}]`")
                        else:
                            st.caption("No explicit sentences found matching linguistic cue patterns. Note: This does not prove the paper omits this topic.")

            with col_b:
                for cat in categories[half:]:
                    info = dims[cat]
                    with st.expander(f"📌 {cat} — Status: {info.get('status', 'Unknown')}", expanded=True):
                        st.markdown(f"**Summary:** {info.get('summary', 'N/A')}")
                        if info.get("evidence"):
                            st.markdown("**Verbatim Source Evidence:**")
                            for ev in info["evidence"]:
                                st.markdown(f"- *\"{ev['text']}\"* `[Page {ev.get('page_number', 1)}]`")
                        else:
                            st.caption("No explicit sentences found matching linguistic cue patterns. Note: This does not prove the paper omits this topic.")


# ==================================================
# SECTION 3: RESEARCH GAP EXPLORER
# ==================================================

elif navigation == "🔍 Research Gap Explorer":
    st.subheader("Research Gap Explorer")
    st.write(
        "Synthesizes author-stated limitations and future-work directions across the corpus. "
        "Clusters overlapping limitations using TF-IDF and cosine similarity to discover candidate open problems."
    )

    if not papers:
        st.info("Upload papers in the **Workspace** tab or load demo papers to discover research gaps.")
    else:
        gap_results = discover_research_gaps(papers)
        candidate_gaps = gap_results.get("candidate_gaps", [])
        explicit_statements = gap_results.get("explicit_statements", [])

        # Prominent Scientific Disclaimer
        st.warning(f"⚠️ **Scope Notice:** {gap_results.get('disclaimer')}")

        st.markdown("### 🧩 Synthesized Research Gap Clusters")

        if candidate_gaps:
            for idx, gap in enumerate(candidate_gaps, start=1):
                is_cross = gap.get("is_inferred", False)
                badge_class = "badge-red" if is_cross else "badge-amber"
                cat_label = gap.get("gap_category", "Identified Constraint")

                st.markdown(f"""
                <div class="paper-card">
                    <span class="badge-tag {badge_class}">{cat_label}</span>
                    <span class="badge-tag badge-blue">Affected Papers: {gap.get('paper_count', 1)}</span>
                    <h4 style="margin-top: 10px; margin-bottom: 8px;">Gap #{idx}: {gap.get('theme')}</h4>
                </div>
                """, unsafe_allow_html=True)

                with st.expander(f"View Supporting Excerpts & Provenance for Gap #{idx} ({gap.get('evidence_count', 1)} Excerpts)", expanded=True):
                    for stmt in gap.get("statements", []):
                        badge_type = "badge-red" if "Limitation" in stmt.get("type", "") else "badge-green"
                        st.markdown(f"""
                        - <span class="badge-tag {badge_type}">{stmt.get('type')}</span>
                          *\"{stmt.get('text')}\"* 
                          — **{stmt.get('paper_name')}**, `Page {stmt.get('page_number')}`
                        """, unsafe_allow_html=True)
        else:
            st.info("No research gaps or limitations were explicitly extracted from the documents.")

        st.markdown("---")
        st.markdown("### 📋 All Explicit Author-Stated Limitations & Future Work")
        st.caption("Direct verbatim excerpts extracted from the uploaded collection with page references.")

        if explicit_statements:
            exp_table = []
            for s in explicit_statements:
                exp_table.append({
                    "Statement Type": s.get("type"),
                    "Source Document": s.get("paper_name"),
                    "Page": s.get("page_number"),
                    "Author-Stated Excerpt": s.get("text")
                })
            st.dataframe(pd.DataFrame(exp_table), use_container_width=True)
        else:
            st.info("No explicit statements found.")


# ==================================================
# SECTION 4: EVIDENCE KNOWLEDGE GRAPH
# ==================================================

elif navigation == "🕸️ Evidence Knowledge Graph":
    st.subheader("Evidence-Linked Research Knowledge Graph")
    st.write(
        "Interactive relational network built with NetworkX and Plotly. "
        "Represents Papers, Methods, Datasets, and Constraints connected exclusively through verified textual evidence."
    )

    if not papers:
        st.info("Upload papers in the **Workspace** tab or load demo papers to construct the knowledge graph.")
    else:
        graph = st.session_state.get("knowledge_graph")
        if graph is None or graph.number_of_nodes() < 2:
            graph = build_multipaper_knowledge_graph(papers)
            st.session_state["knowledge_graph"] = graph

        metrics = get_graph_metrics(graph)

        # Graph Metrics
        g1, g2, g3, g4 = st.columns(4)
        with g1:
            st.metric("Total Nodes", metrics["nodes"])
        with g2:
            st.metric("Total Relationships", metrics["edges"])
        with g3:
            st.metric("Graph Density", metrics["density"])
        with g4:
            top_node = metrics["top_central_nodes"][0][0] if metrics["top_central_nodes"] else "None"
            st.metric("Highest Centrality Concept", top_node)

        # Category Legend
        st.markdown("""
        **Node Category Legend:**
        <span class="badge-tag" style="background:#4f46e5; color:white;">Paper Node</span>
        <span class="badge-tag" style="background:#059669; color:white;">Methodology</span>
        <span class="badge-tag" style="background:#d97706; color:white;">Dataset / Benchmark</span>
        <span class="badge-tag" style="background:#e11d48; color:white;">Limitation</span>
        <span class="badge-tag" style="background:#8b5cf6; color:white;">Key Concept</span>
        """, unsafe_allow_html=True)

        # Plotly Interactive Figure
        fig = render_plotly_graph(graph)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Fallback to matplotlib if plotly unavailable
            fig_mpl = render_matplotlib_graph(graph)
            if fig_mpl:
                st.pyplot(fig_mpl)
            else:
                st.warning("Too few concepts extracted to construct a 2D network layout.")

        st.markdown("---")
        st.subheader("🔍 Node Evidence Inspector")
        all_nodes = sorted(list(graph.nodes()))
        if all_nodes:
            inspected = st.selectbox("Select any graph node to inspect supporting evidence", all_nodes)
            node_data = graph.nodes[inspected]
            n_type = node_data.get("type", "Concept")
            n_evidence = node_data.get("evidence", [])

            st.markdown(f"**Node Name:** {inspected} &nbsp;|&nbsp; **Category:** `{n_type}` &nbsp;|&nbsp; **Network Degree:** `{graph.degree[inspected]}`")

            neighbors = list(graph.neighbors(inspected))
            if neighbors:
                st.markdown(f"**Connected Nodes ({len(neighbors)}):** " + ", ".join([f"`{n}`" for n in neighbors]))

            if n_evidence:
                st.markdown("**Supporting Text Evidence & Provenance:**")
                for ev in n_evidence:
                    st.markdown(f"- *\"{ev.get('text')}\"* `[{ev.get('paper_name')}, Page {ev.get('page_number', 1)}]`")
            else:
                st.caption("No specific evidence excerpts attached to this node.")


# ==================================================
# SECTION 5: ASK THE LITERATURE
# ==================================================

elif navigation == "💬 Ask the Literature":
    st.subheader("Ask the Literature")
    st.write(
        "Query across all indexed papers using sublinear TF-IDF retrieval. "
        "Returns ranked passages with source citations and page provenance. LitSense never fabricates ungrounded answers."
    )

    if not papers:
        st.info("Upload papers in the **Workspace** tab or load demo papers to enable cross-literature search.")
    else:
        search_engine = st.session_state.get("search_engine")
        if search_engine is None or search_engine.matrix is None:
            search_engine = MultiPaperSearchEngine()
            search_engine.index_papers(papers)
            st.session_state["search_engine"] = search_engine

        # Quick Example Buttons
        st.markdown("##### 💡 Example Questions:")
        e1, e2, e3, e4 = st.columns(4)
        sample_q = ""
        if e1.button("Benchmarks & Datasets", use_container_width=True):
            sample_q = "What datasets, benchmarks, or experimental corpora were evaluated?"
        if e2.button("Primary Limitations", use_container_width=True):
            sample_q = "What are the primary limitations, bottlenecks, or failure modes?"
        if e3.button("Proposed Method", use_container_width=True):
            sample_q = "What methodology, architecture, or algorithm is proposed?"
        if e4.button("Future Research", use_container_width=True):
            sample_q = "What future directions or open research questions are suggested?"

        query_input = st.text_input(
            "Enter research question or keywords:",
            value=sample_q if sample_q else "",
            placeholder="e.g., How does attention memory scale with sequence length?"
        )

        top_k = st.slider("Number of ranked passages to retrieve", min_value=3, max_value=12, value=5)

        if query_input.strip():
            with st.spinner("Searching indexed research corpus..."):
                results = search_engine.search(query_input, top_k=top_k)

            if results:
                st.markdown(f"#### Top {len(results)} Evidence Passages Ranked by TF-IDF Relevance:")

                # Evidence-based synthesis note
                st.info(
                    "📌 **Evidence-Based Transparency:** The following passages are directly retrieved from your papers "
                    "using vector cosine similarity ranking. LitSense presents source evidence without generative hallucination."
                )

                for idx, r in enumerate(results, start=1):
                    score_pct = int(r["score"] * 100)
                    st.markdown(f"""
                    <div class="paper-card">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                            <span class="badge-tag badge-blue">Rank #{idx}</span>
                            <span class="badge-tag badge-green">Cosine Relevance: {r['score']:.4f}</span>
                        </div>
                        <p style="font-size: 15px; font-style: italic; color: #1e293b; margin-bottom: 10px;">
                            \"{r['sentence']}\"
                        </p>
                        <div style="font-size: 13px; color: #64748b; font-weight: 600;">
                            📄 Document: <span style="color:#3730a3;">{r['paper_name']}</span> &nbsp;|&nbsp; 📖 Page: <span style="color:#3730a3;">{r['page_number']}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("No relevant passages met the relevance threshold for this query. Try broader academic keywords.")


# ==================================================
# SECTION 6: CLAIM ALIGNMENT (AGREEMENT & DIVERGENCE)
# ==================================================

elif navigation == "⚖️ Claim Alignment (Agreement & Divergence)":
    st.subheader("Cross-Paper Claim Alignment & Stance Discovery")
    st.write(
        "Identifies related findings and claims across different papers using lexical-semantic similarity and stance cues. "
        "Helps researchers locate areas of concordance or potential differing perspectives."
    )

    if len(papers) < 2:
        st.info("Claim alignment requires at least two research papers. Please upload additional papers in the **Workspace** tab or load the 3 demo papers.")
    else:
        claims_analysis = analyze_cross_paper_claims(papers)
        pairs = claims_analysis.get("pairs", [])

        # Transparent scientific disclaimer
        st.warning(f"⚠️ **Viva Defense Note:** {claims_analysis.get('disclaimer')}")

        if pairs:
            st.markdown(f"### Found {len(pairs)} Cross-Paper Claim Pairs:")

            for idx, p in enumerate(pairs, start=1):
                badge_style = "badge-green" if "Concordance" in p["label"] else ("badge-red" if "Divergence" in p["label"] else "badge-blue")

                st.markdown(f"""
                <div class="paper-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                        <span class="badge-tag {badge_style}">{p['label']}</span>
                        <span class="badge-tag badge-amber">Semantic Overlap: {p['similarity']}</span>
                    </div>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 10px;">
                        <div style="background: #f8fafc; padding: 14px; border-radius: 10px; border-left: 4px solid #4f46e5;">
                            <div style="font-size: 12px; font-weight: 700; color: #4338ca; margin-bottom: 4px;">
                                {p['paper_a']} [Page {p['page_a']}]
                            </div>
                            <div style="font-size: 14px; font-style: italic; color: #334155;">\"{p['text_a']}\"</div>
                        </div>
                        <div style="background: #f8fafc; padding: 14px; border-radius: 10px; border-left: 4px solid #059669;">
                            <div style="font-size: 12px; font-weight: 700; color: #047857; margin-bottom: 4px;">
                                {p['paper_b']} [Page {p['page_b']}]
                            </div>
                            <div style="font-size: 14px; font-style: italic; color: #334155;">\"{p['text_b']}\"</div>
                        </div>
                    </div>
                    <div style="margin-top: 10px; font-size: 12px; color: #64748b;">
                        <b>Detection Rationale:</b> {p['rationale']} — <i>{p['status']}</i>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No cross-paper claims exceeded the minimum semantic overlap threshold for pairwise alignment.")


# ==================================================
# SECTION 7: NLP METHODOLOGY
# ==================================================

elif navigation == "📚 NLP Methodology":
    st.subheader("NLP Methodology & Architecture Specification")
    st.write(
        "Comprehensive viva defense documentation detailing all natural language processing algorithms, "
        "libraries, heuristics, and engineering decisions implemented in LitSense V1."
    )

    t1, t2, t3 = st.tabs(["Pipeline Architecture", "Implemented NLP Modules", "Viva Defense & Scope"])

    with t1:
        st.markdown("### End-to-End Processing Flow")
        st.markdown("""
        ```mermaid
        graph TD
            PDF[Uploaded PDF Papers] --> PyMuPDF[PyMuPDF Page Extractor]
            PyMuPDF --> Cleaner[Text Preprocessor & Hyphen Fixer]
            Cleaner --> SpaCy[spaCy Sentence Segmenter & NER]
            SpaCy --> Dimensions[Dimension Extractor: Cues & Provenance]
            Dimensions --> TFIDF[scikit-learn TF-IDF Vectorizer]
            TFIDF --> SearchEngine[MultiPaperSearchEngine]
            TFIDF --> GapCluster[Gap Clusterer: Cosine Similarity]
            Dimensions --> NetX[NetworkX Heterogeneous Graph]
            NetX --> PlotlyVis[Plotly 2D Interactive Network]
        ```
        """)

    with t2:
        st.markdown("### Concrete Implementation Details")
        st.markdown("""
| Pipeline Stage | Python Library | Code Module | Purpose & Technique |
| :--- | :--- | :--- | :--- |
| **PDF Text Ingestion** | `pymupdf` (v2.0+) | `processor.py:extract_pdf_pages` | Extracts UTF-8 text streams page-by-page. Preserves page boundaries and detects empty text layers in scanned PDFs. |
| **Text Preprocessing** | `re` (Standard Lib) | `processor.py:clean_text` | Normalizes irregular linebreaks, strips carriage returns, and rejoins hyphenated line breaks (`algo-\\nrithm` → `algorithm`). |
| **Sentence Segmentation** | `spacy` (`en_core_web_sm`) | `processor.py:split_sentences` | Dependency-parsed sentence boundary detection. Falls back to regex boundary rules if spaCy is unavailable. |
| **Dimension Extraction** | Rule-Based Linguistic Cues | `processor.py:extract_paper_dimensions` | Pattern matching against Problem, Method, Dataset, Finding, and Limitation cues. Weights multi-word phrases over single tokens. |
| **Information Retrieval** | `scikit-learn` (`TfidfVectorizer`) | `search_engine.py:MultiPaperSearchEngine` | Sublinear term frequency weighting with unigram and bigram features. Evaluates query-document cosine similarities. |
| **Cross-Paper Gap Clustering** | `scikit-learn` (`cosine_similarity`) | `gap_analyzer.py:discover_research_gaps` | Agglomerative threshold clustering (`threshold=0.20`) of author-stated limitations across papers to find convergent themes. |
| **Knowledge Graph** | `networkx` & `plotly` | `knowledge_graph.py` | Builds a heterogeneous graph of Papers, Methods, Datasets, and Limitations with verified evidence citations. |
| **Claim Stance Discovery** | Lexical Stance Patterns | `gap_analyzer.py:analyze_cross_paper_claims` | Identifies related cross-paper claims and categorizes potential concordance vs divergence hypotheses. |
        """)

    with t3:
        st.markdown("### Viva Defense Q&A & Scope Transparency")
        
        with st.expander("Q: Why use TF-IDF and rule-based NLP instead of an external LLM API?", expanded=True):
            st.write(
                "1. **Zero Fabrication / Hallucination:** In literature reviews, hallucinated citations or invented claims disqualify academic research. LitSense only quotes extracted sentences.\n"
                "2. **Zero API Cost & Data Privacy:** Works completely offline on student laptops without requiring OpenAI or cloud API keys.\n"
                "3. **Explainability:** TF-IDF weights, cosine similarity matrices, and linguistic cues are mathematically transparent and easy to justify in an academic defense."
            )

        with st.expander("Q: How does LitSense preserve evidence traceability?", expanded=True):
            st.write(
                "Every sentence extracted by `processor.py` is tagged with a dictionary containing `paper_name` and `page_number`. "
                "This tuple is propagated across the comparison matrix, search engine, gap explorer, and knowledge graph."
            )

        with st.expander("Q: What are the primary limitations of V1 and future enhancements?", expanded=True):
            st.write(
                "1. **Scanned PDFs:** Requires OCR (e.g., Tesseract or PyMuPDF OCR) for image-only scans.\n"
                "2. **Natural Language Inference:** Current claim alignment relies on lexical overlap and stance markers; future work can integrate local cross-encoders (e.g. DeBERTa-v3-NLI).\n"
                "3. **Dense Semantic Embeddings:** Can be augmented with local sentence-transformers (e.g., BGE or SciBERT) for deeper semantic synonym matching."
            )


# --------------------------------------------------
# 8. FOOTER
# --------------------------------------------------

st.markdown("---")
st.caption(
    "LitSense V1 • Intelligent Literature Review and Research Gap Discovery • "
    "Designed for academic integrity and verifiable literature synthesis."
)
