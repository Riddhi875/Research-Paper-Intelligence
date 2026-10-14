import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx
import re
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from processor import (
    extract_text,
    summarize_text,
    split_sentences,
    create_study_notes
)
from knowledge_graph import build_knowledge_graph
from search_engine import ResearchPaperSearch


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="PaperPilot",
    
    layout="wide"
)

# --------------------------------------------------
# MODERN UI / UX DESIGN
# --------------------------------------------------

st.markdown("""
<style>

/* Main application background */
.stApp {
    background-color: #f5f7fc;
    color: black;
}

/* Main content width and spacing */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

/* Hide Streamlit branding */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

/* Main headings */
h1, h2, h3 {
    color: #172554 !important;
    letter-spacing: -0.5px;
}

/* Hero section */
.hero-card {
    background: linear-gradient(120deg, #172554, #4338ca);
    padding: 34px;
    border-radius: 22px;
    color: white !important;
    margin-bottom: 28px;
    box-shadow: 0 10px 30px rgba(30, 41, 99, 0.14);
}

.hero-card h1 {
    color: white !important;
    font-size: 34px;
    margin-bottom: 8px;
}

.hero-card p {
    color: #e0e7ff !important;
    font-size: 16px;
    margin-bottom: 0;
}

/* Section cards */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 16px;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #4f46e5, #6366f1);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.65rem 1.2rem;
    font-weight: 600;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    background: #3730a3;
    color: white;
    border: none;
    box-shadow: 0 5px 15px rgba(79, 70, 229, 0.2);
}

/* File uploader */
div[data-testid="stFileUploader"] {
    background: white;
    padding: 18px;
    border: 1px dashed #a5b4fc;
    border-radius: 16px;
}

/* Tabs */
button[data-baseweb="tab"] {
    font-weight: 600;
    padding: 12px 16px;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #4338ca;
}

/* Metrics */
div[data-testid="stMetric"] {
    background: white;
    padding: 20px;
    border: 1px solid #e5e7eb;
    border-radius: 16px;
    box-shadow: 0 3px 12px rgba(15, 23, 42, 0.035);
}

div[data-testid="stMetricLabel"] {
    color: #64748b;
}

div[data-testid="stMetricValue"] {
    color: #3730a3;
    font-weight: 700;
}

/* Expandable sections */
details {
    background: white;
    border: 1px solid #e5e7eb !important;
    border-radius: 12px !important;
    margin-bottom: 10px;
    padding: 5px 10px;
}

/* Text inputs */
.stTextInput input,
.stTextArea textarea {
    border-radius: 10px;
}

/* Informational messages */
div[data-testid="stAlert"] {
    border-radius: 12px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #eef2ff;
    border-right: 1px solid #e0e7ff;
}

/* Divider */
hr {
    border-color: #e2e8f0;
}

/* Small screens */
@media (max-width: 768px) {
    .block-container {
        padding: 1rem;
    }

    .hero-card {
        padding: 22px;
    }

    .hero-card h1 {
        font-size: 27px;
    }
}

</style>
""", unsafe_allow_html=True)

# Main title is already displayed in the PaperPilot hero banner

st.markdown("""
<div class="hero-card">
    <h1> PaperPilot </h1>
    <p>
        Turn complex research papers into organized notes,
        connected concepts and searchable knowledge.
    </p>
</div>
""", unsafe_allow_html=True)



# --------------------------------------------------
# 2. HELPER FUNCTIONS
# --------------------------------------------------

def extract_key_terms(text, limit=20):
    """Extract frequent, meaningful words and noun phrases."""

    words = re.findall(r"\b[a-zA-Z][a-zA-Z-]{2,}\b", text.lower())

    filtered_words = [
        word for word in words
        if word not in ENGLISH_STOP_WORDS
        and not word.isdigit()
    ]

    word_counts = Counter(filtered_words)

    # Include meaningful noun phrases from spaCy
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text[:500000])

        phrases = []

        for chunk in doc.noun_chunks:
            phrase = " ".join(chunk.text.lower().split())

            if (
                len(phrase) >= 4
                and len(phrase.split()) <= 5
                and not all(
                    word in ENGLISH_STOP_WORDS
                    for word in phrase.split()
                )
            ):
                phrases.append(phrase)

        phrase_counts = Counter(phrases)

    except Exception:
        phrase_counts = Counter()

    return word_counts, phrase_counts


def display_evidence(sentences, empty_message):
    """Display source sentences as evidence."""

    if sentences:
        for sentence in sentences:
            st.markdown(f"- {sentence}")
    else:
        st.info(empty_message)


with st.sidebar:
    st.markdown("## PaperPilot")
    st.caption("Your intelligent research companion")

    st.divider()

    st.markdown("### Your workspace")
    st.write("📄 PDF text extraction")
    st.write("📝 Extractive summarization")
    st.write("💡 Key concept exploration")
    st.write("🕸️ Knowledge graph")
    st.write("🔎 TF-IDF smart search")

    st.divider()

    st.markdown("### About")
    st.caption(
        "An NLP-based research assistant for exploring "
        "academic papers and organizing study material."
    )


# --------------------------------------------------
# 3. PDF UPLOAD
# --------------------------------------------------


st.subheader("Start with your research paper")

st.write(
    "Upload a PDF to extract its content and explore its key findings."
)

uploaded_file = st.file_uploader(
    "Choose a research paper",
    type=["pdf"],
    help="Upload a text-based PDF research paper."
)


if uploaded_file is None:
    st.info("Upload a PDF to begin analyzing your research paper.")

    st.markdown("### What you can explore")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**📝 Summary and Notes**")
        st.write("Review important sentences and organized study notes.")

    with col2:
        st.markdown("**🔎 Concepts and Search**")
        st.write("Explore important terms and find relevant passages.")

    with col3:
        st.markdown("**🕸️ Knowledge Graph**")
        st.write("Explore concepts and their co-occurrence connections.")

    st.stop()


# --------------------------------------------------
# 4. ANALYZE BUTTON
# --------------------------------------------------

if st.button("Analyze Research Paper", type="primary"):

    with st.spinner("Reading and analyzing the PDF..."):

        try:
            pdf_bytes = uploaded_file.getvalue()

            text = extract_text(pdf_bytes)

            if not text.strip():
                st.error(
                    "No readable text was extracted. "
                    "This PDF may contain scanned pages. "
                    "OCR may be needed for scanned PDFs."
                )
                st.stop()

            sentences = split_sentences(text)

            if not sentences:
                st.error(
                    "The paper was opened, but no usable sentences "
                    "were extracted."
                )
                st.stop()

            summary = summarize_text(text, max_sentences=7)
            study_notes = create_study_notes(text)
            graph = build_knowledge_graph(text, max_concepts=12)
            st.subheader("📊 Most Frequent Concepts")

            if graph.number_of_nodes() > 0:
                concept_data = sorted(
                    graph.nodes(data=True),
                    key=lambda item: item[1].get("frequency", 0),
                    reverse=True
                )[:10]

                concepts = [node for node, data in concept_data]
                frequencies = [
                    data.get("frequency", 0)
                    for node, data in concept_data
                ]

                fig, ax = plt.subplots(figsize=(10, 5))
                ax.barh(concepts[::-1], frequencies[::-1])
                ax.set_xlabel("Frequency")
                ax.set_title("Top 10 Concepts in the Research Paper")
                ax.grid(axis="x", alpha=0.25)
                fig.tight_layout()

                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("No concepts were found in this paper.")

            st.subheader("🔥 Concept Relationship Heatmap")

            if graph.number_of_nodes() > 1:
                top_nodes = [
                    node for node, data in sorted(
                        graph.nodes(data=True),
                        key=lambda item: item[1].get("frequency", 0),
                        reverse=True
                    )[:12]
                ]

                matrix = nx.to_numpy_array(
                    graph,
                    nodelist=top_nodes,
                    weight="weight"
                )

                fig, ax = plt.subplots(figsize=(10, 7))

                image = ax.imshow(matrix, cmap="Blues")
                ax.set_xticks(range(len(top_nodes)))
                ax.set_yticks(range(len(top_nodes)))
                ax.set_xticklabels(top_nodes, rotation=45, ha="right")
                ax.set_yticklabels(top_nodes)

                ax.set_title("How Frequently Concepts Are Connected")
                fig.colorbar(image, ax=ax, label="Connection strength")
                fig.tight_layout()

                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("At least two concepts are needed for a heatmap.")

            searcher = ResearchPaperSearch()
            searcher.index_text(text)

            word_counts, phrase_counts = extract_key_terms(text)

            # Save results in session state so tabs and search
            # remain available after Streamlit reruns.
            st.session_state["paper_text"] = text
            st.session_state["paper_sentences"] = sentences
            st.session_state["paper_summary"] = summary
            st.session_state["paper_notes"] = study_notes
            st.session_state["paper_graph"] = graph
            st.session_state["paper_searcher"] = searcher
            st.session_state["paper_word_counts"] = word_counts
            st.session_state["paper_phrase_counts"] = phrase_counts
            st.session_state["paper_name"] = uploaded_file.name

            st.success("Analysis completed successfully!")

        except Exception as error:
            st.error("The paper could not be analyzed.")
            st.exception(error)


# --------------------------------------------------
# 5. DISPLAY ANALYSIS RESULTS
# --------------------------------------------------

if "paper_text" not in st.session_state:
    st.caption("Click **Analyze Research Paper** after uploading your PDF.")
    st.stop()

text = st.session_state["paper_text"]
sentences = st.session_state["paper_sentences"]
summary = st.session_state["paper_summary"]
study_notes = st.session_state["paper_notes"]
graph = st.session_state["paper_graph"]
searcher = st.session_state["paper_searcher"]
word_counts = st.session_state["paper_word_counts"]
phrase_counts = st.session_state["paper_phrase_counts"]

st.divider()


st.subheader(st.session_state.get("paper_name", "Research Paper"))

st.markdown("## Document Overview")

metric1, metric2, metric3 = st.columns(3)

with metric1:
    st.metric(
        label="📄 Total Words",
        value=f"{len(text.split()):,}"
    )

with metric2:
    st.metric(
        label="📝 Sentences",
        value=f"{len(sentences):,}"
    )

with metric3:
    st.metric(
        label="🕸️ Key Concepts",
        value=graph.number_of_nodes()
    )

st.divider()



# --------------------------------------------------
# 6. NAVIGATION TABS
# --------------------------------------------------

tabs = st.tabs([
    "📝 Summary",
    "📖 Study Notes",
    "💡 Key Concepts",
    "📄 Full Text",
    "🔍 Find in Paper",
    "🕸️ Knowledge Graph",
    "🤖 Smart Search"
])


# --------------------------------------------------
# TAB 1: SUMMARY
# --------------------------------------------------

with tabs[0]:

    st.header("Research Paper Summary")

    st.caption(
        "These are important sentences selected from the paper. "
        "They are extracted, not rewritten by a generative AI model."
    )

    if summary:
        for index, sentence in enumerate(summary, start=1):
            st.markdown(f"**{index}.** {sentence}")
    else:
        st.info("No summary sentences were available.")

    st.divider()

    st.subheader("Paper at a Glance")

    st.write(
        "The following information is extracted from the document. "
        "It is not a separately verified interpretation of the paper."
    )

    st.write(f"**Document:** {st.session_state.get('paper_name', 'PDF')}")

    st.write(f"**Word count:** {len(text.split()):,}")

    st.write(f"**Sentence count:** {len(sentences):,}")


# --------------------------------------------------
# TAB 2: STUDY NOTES
# --------------------------------------------------

with tabs[1]:

    st.header("Study Notes")

    st.write(
        "Use these sections to review the paper. Sentences are grouped "
        "using keyword matches, so check the original context when "
        "revising important research findings."
    )

    for section, section_sentences in study_notes.items():

        with st.expander(
            section,
            expanded=(section == "Paper overview")
        ):

            if section_sentences:
                for sentence in section_sentences:
                    st.markdown(f"- {sentence}")
            else:
                st.info(
                    "No sentences matched this section's keywords. "
                    "The paper may discuss this topic using different wording."
                )

    st.divider()

    st.subheader("Additional Reading Sentences")

    st.write(
        "Review more sentences from the paper beyond the selected summary."
    )

    number_to_show = st.slider(
        "Number of sentences to display",
        min_value=10,
        max_value=min(100, max(10, len(sentences))),
        value=min(30, max(10, len(sentences))),
        step=10
    )

    for index, sentence in enumerate(
        sentences[:number_to_show],
        start=1
    ):
        st.markdown(f"**{index}.** {sentence}")


# --------------------------------------------------
# TAB 3: KEY CONCEPTS
# --------------------------------------------------

with tabs[2]:

    st.header("Key Concepts")

    st.write(
        "Explore frequent terms and phrases from the paper. "
        "Frequency indicates how often a term appears; it does not "
        "necessarily indicate scientific importance."
    )

    phrase_items = [
        (phrase, count)
        for phrase, count in phrase_counts.most_common(30)
    ]

    word_items = [
        (word, count)
        for word, count in word_counts.most_common(30)
    ]

    concept_mode = st.radio(
        "Concept type",
        ["Meaningful phrases", "Individual words"],
        horizontal=True
    )

    items = (
        phrase_items
        if concept_mode == "Meaningful phrases"
        else word_items
    )

    if items:

        search_term = st.text_input(
            "Filter concepts",
            placeholder="Type a concept to find it..."
        )

        filtered_items = [
            (term, count)
            for term, count in items
            if search_term.lower() in term.lower()
        ]

        for term, count in filtered_items:

            with st.expander(term.title()):

                st.write(f"**Occurrences:** {count}")

                # Search for sentences that contain the concept.
                matching_sentences = [
                    sentence
                    for sentence in sentences
                    if term.lower() in sentence.lower()
                ][:3]

                st.markdown("**Context in the paper**")

                display_evidence(
                    matching_sentences,
                    "No matching sentence was found in the extracted text."
                )

                # Show graph relationships when this term is a graph node.
                normalized_term = term.lower()

                if normalized_term in graph:

                    neighbors = list(graph.neighbors(normalized_term))

                    st.markdown("**Related graph concepts**")

                    if neighbors:
                        st.write(", ".join(neighbors[:5]))
                    else:
                        st.write("No strong connections were identified.")

    else:
        st.info("No concepts were extracted.")


# --------------------------------------------------
# TAB 4: FULL TEXT
# --------------------------------------------------

with tabs[3]:

    st.header("Full Extracted Text")

    st.caption(
        "This is text extracted from the PDF. Original formatting, "
        "tables, figures, and equations may not be preserved perfectly."
    )

    st.download_button(
        label="Download Extracted Text",
        data=text,
        file_name="extracted_research_paper.txt",
        mime="text/plain"
    )

    with st.expander("View complete extracted text", expanded=True):
        st.text_area(
            "Extracted PDF content",
            value=text,
            height=600,
            disabled=True,
            label_visibility="collapsed"
        )


# --------------------------------------------------
# TAB 5: FIND IN PAPER
# --------------------------------------------------

with tabs[4]:

    st.header("Find in Paper")

    query = st.text_input(
        "Search for a word or phrase",
        key="find_in_paper_query",
        placeholder="Example: neural networks"
    )

    if query.strip():

        matching_sentences = [
            sentence
            for sentence in sentences
            if query.lower() in sentence.lower()
        ]

        st.write(
            f"Found **{len(matching_sentences)}** matching sentences."
        )

        if matching_sentences:

            for index, sentence in enumerate(
                matching_sentences[:100],
                start=1
            ):
                st.markdown(f"**{index}.** {sentence}")

            if len(matching_sentences) > 100:
                st.caption(
                    "Only the first 100 matching sentences are displayed."
                )

        else:
            st.info(
                "No exact matches found. Try a different word or phrase."
            )


# --------------------------------------------------
# TAB 6: KNOWLEDGE GRAPH
# --------------------------------------------------

with tabs[5]:

    st.header("Knowledge Graph")

    st.write(
        "Each node represents an extracted concept. A line means the "
        "two concepts appeared together in at least two sentences. "
        "This does not prove a causal or direct scientific relationship."
    )

    if graph.number_of_nodes() == 0:

        st.info("No concepts were extracted from this paper.")

    else:

        max_visible = min(12, graph.number_of_nodes())

        selected_limit = st.slider(
            "Number of concepts to show",
            min_value=3 if max_visible >= 3 else 1,
            max_value=max_visible,
            value=max_visible,
            key="graph_concept_limit"
        )

        top_nodes = sorted(
            graph.nodes(data=True),
            key=lambda item: item[1].get("frequency", 1),
            reverse=True
        )[:selected_limit]

        visible_nodes = [node for node, data in top_nodes]

        display_graph = graph.subgraph(visible_nodes).copy()

        if display_graph.number_of_nodes() == 1:
            positions = {visible_nodes[0]: (0, 0)}
        else:
            positions = nx.spring_layout(
                display_graph,
                seed=42,
                k=1.5,
                iterations=100
            )

        fig, ax = plt.subplots(figsize=(14, 10))

        node_sizes = [
            900 + 250 * display_graph.nodes[node].get("frequency", 1)
            for node in display_graph.nodes
        ]

        nx.draw_networkx_nodes(
            display_graph,
            positions,
            node_size=node_sizes,
            alpha=0.85,
            ax=ax
        )

        nx.draw_networkx_edges(
            display_graph,
            positions,
            width=1.2,
            alpha=0.4,
            ax=ax
        )

        nx.draw_networkx_labels(
            display_graph,
            positions,
            font_size=8,
            font_weight="bold",
            ax=ax
        )

        ax.set_title("Key Concepts and Co-occurrence Connections")
        ax.axis("off")

        st.pyplot(fig)
        plt.close(fig)

        st.divider()

        st.subheader("Explore a Concept")

        selected_concept = st.selectbox(
            "Select a concept",
            visible_nodes,
            key="selected_graph_concept"
        )

        details = graph.nodes[selected_concept]

        st.markdown(f"**Concept:** {selected_concept.title()}")

        st.write(
            f"**Occurrences in extracted phrases:** "
            f"{details.get('frequency', 1)}"
        )

        st.markdown("**Supporting sentences**")

        display_evidence(
            details.get("evidence", []),
            "No supporting sentences were saved for this concept."
        )

        related = sorted(
            graph[selected_concept].items(),
            key=lambda item: item[1].get("weight", 1),
            reverse=True
        )

        st.markdown("**Related concepts**")

        if related:

            for related_concept, edge_data in related[:5]:

                st.markdown(
                    f"- **{related_concept}** — appeared together in "
                    f"{edge_data.get('weight', 1)} sentence(s)"
                )

        else:
            st.info(
                "No strong co-occurrence connections were found. "
                "This concept may still be important to the paper."
            )


# --------------------------------------------------
# TAB 7: SMART SEARCH
# --------------------------------------------------

with tabs[6]:

    st.header("Smart Search")

    st.write(
        "Search for a question or topic to find relevant passages. "
        "Results depend on the search method implemented in "
        "search_engine.py."
    )

    smart_query = st.text_input(
        "What do you want to find?",
        key="smart_search_query",
        placeholder="Example: What method does the paper propose?"
    )

    if st.button("Search Paper", key="smart_search_button"):

        if not smart_query.strip():

            st.warning("Enter a search query first.")

        else:

            with st.spinner("Searching the paper..."):

                try:
                    
                    if searcher is None:
                        st.warning(
                            "Search is unavailable. Please analyze the PDF again."
                        )
                    else:
                        results = searcher.search(smart_query)

                        if results:
                            for index, result in enumerate(results, start=1):
                                st.markdown(f"**Result {index}**")
                                st.write(result["sentence"])
                                st.caption(
                                    f"Relevance score: {result['score']:.3f}"
                                )
                                st.divider()
                        else:
                            st.info(
                                "No matching passages found. Try different keywords."
                            )


                    st.session_state["smart_search_results"] = results
                    st.session_state["smart_search_last_query"] = smart_query

                except Exception as error:
                    st.error("Search failed.")
                    st.exception(error)

    if "smart_search_results" in st.session_state:

        results = st.session_state["smart_search_results"]

        st.subheader("Search Results")

        if results:

            if isinstance(results, str):
                st.write(results)

            elif isinstance(results, list):

                for index, result in enumerate(results, start=1):

                    with st.container():

                        if isinstance(result, dict):

                            content = (
                                result.get("sentence")
                                or result.get("text")
                                or result.get("content")
                                or str(result)
                            )

                            score = result.get("score")

                            st.markdown(f"**Result {index}**")

                            if score is not None:
                                st.caption(f"Relevance score: {score}")

                            st.write(content)

                        else:
                            st.markdown(f"**Result {index}**")
                            st.write(result)

                        st.divider()

            else:
                st.write(results)

        else:
            st.info(
                "No relevant passages were returned. "
                "Try a shorter or more specific query."
            )


# --------------------------------------------------
# 7. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Research Paper Intelligence | NLP-based text extraction, "
    "extractive summarization, keyword analysis, and concept "
    "co-occurrence graph. Automatically extracted results should "
    "be checked against the original paper."
)
