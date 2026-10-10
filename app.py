import matplotlib.pyplot as plt
import networkx as nx

from knowledge_graph import build_knowledge_graph
from search_engine import ResearchPaperSearch
import streamlit as st
import pymupdf
from processor import extract_text, summarize_text, split_sentences

st.set_page_config(
    page_title="Research Paper Intelligence",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Research Paper Intelligence")
st.write(
    "Upload a research paper to generate summaries, study notes, "
    "key concepts, and searchable extracted text."
)

uploaded_file = st.file_uploader(
    "Upload your research paper (PDF)",
    type=["pdf"]
)

if uploaded_file is not None:
    st.success(f"Uploaded: {uploaded_file.name}")

    if st.button("🔍 Analyze Research Paper", type="primary"):
        try:
            with st.spinner("Extracting and analyzing your paper..."):
                pdf_bytes = uploaded_file.getvalue()
                text = extract_text(pdf_bytes)

                if not text.strip():
                    st.error(
                        "No text could be extracted. "
                        "This may be a scanned PDF."
                    )
                    st.stop()

                summary = summarize_text(text, max_sentences=5)
                sentences = split_sentences(text)
                graph = build_knowledge_graph(text)

                searcher = ResearchPaperSearch()
                indexed_count = searcher.index_text(text)

                # Basic key-concept extraction
                from collections import Counter
                import re
                from sklearn.feature_extraction.text import (
                    ENGLISH_STOP_WORDS
                )

                words = re.findall(r"\b[a-zA-Z][a-zA-Z-]{2,}\b", text.lower())
                keywords = [
                    word for word in words
                    if word not in ENGLISH_STOP_WORDS
                ]
                top_keywords = Counter(keywords).most_common(15)

            st.success("Analysis completed!")

            tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
                "📝 Summary",
                "📒 Study Notes",
                "🔑 Key Concepts",
                "📄 Full Text",
                "🔎 Find in Paper",
                "🕸️ Knowledge Graph",
                "🔍 Smart Search"
            ])

            with tab1:
                st.header("Research Paper Summary")
                for sentence in summary:
                    st.write("•", sentence)

                st.download_button(
                    "Download Summary",
                    "\n\n".join(summary),
                    file_name="research_paper_summary.txt",
                    mime="text/plain"
                )

            with tab2:
                st.header("Study Notes")
                st.info(
                    "These are automatically generated notes based "
                    "on sentences extracted from the paper."
                )

                for i, sentence in enumerate(sentences[:30], start=1):
                    st.markdown(f"**{i}.** {sentence}")

                notes = "\n\n".join(
                    f"{i}. {sentence}"
                    for i, sentence in enumerate(sentences[:30], start=1)
                )

                st.download_button(
                    "Download Study Notes",
                    notes,
                    file_name="research_paper_notes.txt",
                    mime="text/plain"
                )

            with tab3:
                st.header("Key Concepts and Keywords")
                st.write(
                    "Frequently occurring words, excluding common "
                    "English stop words. These are candidate keywords, "
                    "not verified scientific concepts."
                )

                for word, count in top_keywords:
                    st.write(f"**{word}** — appears {count} times")

            with tab4:
                st.header("Extracted Research Paper Text")
                st.caption(f"Extracted {len(text):,} characters.")

                st.text_area(
                    "Full extracted text",
                    value=text,
                    height=450
                )

                st.download_button(
                    "Download Extracted Text",
                    text,
                    file_name="extracted_research_paper.txt",
                    mime="text/plain"
                )

            with tab5:
                st.header("Find in Paper")

                query = st.text_input("Enter a word or phrase")

                if query:
                    matches = [
                        sentence for sentence in sentences
                        if query.lower() in sentence.lower()
                    ]

                    if matches:
                        st.write(f"Found {len(matches)} matching sentences.")
                        for match in matches:
                            st.write("•", match)
                    else:
                        st.warning("No exact text matches found.")
            with tab6:
                st.header("Research Paper Knowledge Graph")

                st.write(
                    "Nodes represent extracted concepts. "
                    "Edges represent concepts found together in a sentence."
                )

                if graph.number_of_nodes() == 0:
                    st.warning("No concepts were extracted from this paper.")

                else:
                    fig, ax = plt.subplots(figsize=(12, 8))

                    positions = nx.spring_layout(
                        graph,
                        seed=42,
                        k=0.9
                    )

                    node_sizes = [
                        400 + graph.nodes[node].get("frequency", 1) * 100
                        for node in graph.nodes()
                    ]

                    nx.draw_networkx(
                        graph,
                        positions,
                        ax=ax,
                        node_size=node_sizes,
                        font_size=8,
                        with_labels=True
                    )

                    ax.axis("off")
                    st.pyplot(fig)
                    plt.close(fig)

                    st.subheader("Extracted Relationships")

                    edges = [
                        {
                            "Concept 1": first,
                            "Concept 2": second,
                            "Co-occurrences": data.get("weight", 1)
                        }
                        for first, second, data in graph.edges(data=True)
                    ]

                    if edges:
                        st.dataframe(edges, use_container_width=True)
                    else:
                        st.info("No concept relationships were found.")
            with tab7:
                st.header("Search Your Research Paper")

                query = st.text_input(
                    "Ask a question or enter research keywords",
                    key="smart_search_query"
                )

                top_k = st.slider(
                    "Number of results",
                    min_value=1,
                    max_value=10,
                    value=5
                )

                if query:
                    results = searcher.search(query, top_k=top_k)

                    if results:
                        for i, result in enumerate(results, start=1):
                            st.markdown(f"**Result {i}**")
                            st.write(result["sentence"])
                            st.caption(
                                f"Relevance score: {result['score']:.3f}"
                            )
                            st.divider()
                    else:
                        st.info(
                            "No relevant sentences found. "
                            "Try different keywords."
                        )

        except Exception as e:
            st.error("The paper could not be analyzed.")
            st.exception(e)
