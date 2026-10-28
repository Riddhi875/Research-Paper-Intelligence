import unittest
import pymupdf
from processor import extract_pdf_pages, clean_text, split_sentences, extract_paper_dimensions, extract_key_concepts
from search_engine import MultiPaperSearchEngine
from knowledge_graph import build_multipaper_knowledge_graph, render_plotly_graph, get_graph_metrics
from gap_analyzer import discover_research_gaps, analyze_cross_paper_claims


def create_synthetic_pdf(text_pages: list) -> bytes:
    """Create an in-memory PDF using PyMuPDF for reliable unit testing."""
    doc = pymupdf.open()
    for text in text_pages:
        page = doc.new_page()
        page.insert_textbox(pymupdf.Rect(50, 50, 550, 750), text)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


class TestLitSenseCore(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Paper 1 sample content
        cls.p1_pages = [
            "Introduction: In this paper, we address the problem of excessive computational latency in transformer architectures. Traditional approaches suffer from quadratic memory scaling.",
            "Methodology: We propose FlashAttention-Lite, an efficient attention mechanism. We evaluate on the ImageNet benchmark and GLUE dataset.",
            "Results and Limitations: Results demonstrate an accuracy of 92.4%, consistently outperforming baseline models. However, our approach fails when input sequence length exceeds 4096 tokens due to memory constraints. In the future, we plan to investigate sparse attention kernels."
        ]
        cls.pdf1_bytes = create_synthetic_pdf(cls.p1_pages)

        # Paper 2 sample content
        cls.p2_pages = [
            "Abstract: This study explores sparse representations for language processing. The main challenge remains processing long context documents efficiently.",
            "Method: We introduce SparseSeq, a linear complexity neural network architecture tested on the GLUE dataset and SQuAD corpus.",
            "Evaluation: Experimental results show improved performance across all benchmarks. A limitation of our method is that it requires high computational overhead during pretraining. Further research is needed to reduce training FLOPs."
        ]
        cls.pdf2_bytes = create_synthetic_pdf(cls.p2_pages)

    def test_text_cleaning(self):
        dirty = "Transformer-\nbased   models   suffer \r\n from latency."
        cleaned = clean_text(dirty)
        self.assertIn("Transformerbased", cleaned)
        self.assertNotIn("\r", cleaned)
        self.assertNotIn("   ", cleaned)

    def test_pdf_extraction_and_page_provenance(self):
        result = extract_pdf_pages(self.pdf1_bytes, filename="paper1.pdf")
        self.assertEqual(result["status"], "Success")
        self.assertEqual(result["total_pages"], 3)
        self.assertEqual(len(result["pages"]), 3)
        self.assertEqual(result["pages"][0]["page_number"], 1)
        self.assertIn("FlashAttention-Lite", result["pages"][1]["text"])

    def test_empty_or_corrupted_pdf_handling(self):
        # Empty byte stream
        result = extract_pdf_pages(b"not a real pdf", filename="corrupt.pdf")
        self.assertIn("Corrupted", result["status"])
        self.assertEqual(result["total_pages"], 0)

        # Empty pages PDF
        empty_doc = pymupdf.open()
        empty_doc.new_page()  # blank page
        empty_bytes = empty_doc.tobytes()
        empty_doc.close()
        res_empty = extract_pdf_pages(empty_bytes, filename="blank.pdf")
        self.assertEqual(res_empty["status"], "Scanned / OCR Needed (Empty Text Layer)")

    def test_dimension_extraction(self):
        extracted = extract_pdf_pages(self.pdf1_bytes, filename="paper1.pdf")
        dims = extract_paper_dimensions(extracted["pages"], "paper1.pdf")

        # Check problem extraction
        self.assertIn("problem", dims["Problem"]["summary"].lower())
        self.assertTrue(len(dims["Problem"]["evidence"]) > 0)
        self.assertEqual(dims["Problem"]["evidence"][0]["page_number"], 1)

        # Check methodology extraction
        self.assertTrue(len(dims["Methodology"]["evidence"]) > 0)
        self.assertIn("flashattention", dims["Methodology"]["summary"].lower())

        # Check limitations extraction
        self.assertTrue(len(dims["Limitations"]["evidence"]) > 0)
        self.assertIn("fails", dims["Limitations"]["summary"].lower())

    def test_multi_paper_search_engine(self):
        ext1 = extract_pdf_pages(self.pdf1_bytes, filename="paper1.pdf")
        ext2 = extract_pdf_pages(self.pdf2_bytes, filename="paper2.pdf")

        engine = MultiPaperSearchEngine()
        indexed_count = engine.index_papers([ext1, ext2])
        self.assertGreater(indexed_count, 0)

        # Search for GLUE benchmark
        results = engine.search("GLUE dataset benchmark", top_k=5)
        self.assertGreater(len(results), 0)
        top = results[0]
        self.assertIn("sentence", top)
        self.assertIn("paper_name", top)
        self.assertIn("page_number", top)
        self.assertGreater(top["score"], 0.05)

    def test_knowledge_graph_construction(self):
        ext1 = extract_pdf_pages(self.pdf1_bytes, filename="paper1.pdf")
        dims1 = extract_paper_dimensions(ext1["pages"], "paper1.pdf")
        ext1["dimensions"] = dims1
        ext1["key_concepts"] = ["transformer", "latency", "attention"]

        ext2 = extract_pdf_pages(self.pdf2_bytes, filename="paper2.pdf")
        dims2 = extract_paper_dimensions(ext2["pages"], "paper2.pdf")
        ext2["dimensions"] = dims2
        ext2["key_concepts"] = ["sparse", "neural network", "attention"]

        graph = build_multipaper_knowledge_graph([ext1, ext2])
        self.assertGreater(graph.number_of_nodes(), 2)
        self.assertGreater(graph.number_of_edges(), 1)

        metrics = get_graph_metrics(graph)
        self.assertIn("nodes", metrics)
        self.assertIn("edges", metrics)

        fig = render_plotly_graph(graph)
        self.assertIsNotNone(fig)

    def test_gap_discovery(self):
        ext1 = extract_pdf_pages(self.pdf1_bytes, filename="paper1.pdf")
        ext1["dimensions"] = extract_paper_dimensions(ext1["pages"], "paper1.pdf")

        ext2 = extract_pdf_pages(self.pdf2_bytes, filename="paper2.pdf")
        ext2["dimensions"] = extract_paper_dimensions(ext2["pages"], "paper2.pdf")

        gaps = discover_research_gaps([ext1, ext2])
        self.assertIn("candidate_gaps", gaps)
        self.assertIn("explicit_statements", gaps)
        self.assertIn("disclaimer", gaps)
        self.assertGreater(len(gaps["explicit_statements"]), 0)

    def test_cross_paper_claim_alignment(self):
        ext1 = extract_pdf_pages(self.pdf1_bytes, filename="paper1.pdf")
        ext1["dimensions"] = extract_paper_dimensions(ext1["pages"], "paper1.pdf")

        ext2 = extract_pdf_pages(self.pdf2_bytes, filename="paper2.pdf")
        ext2["dimensions"] = extract_paper_dimensions(ext2["pages"], "paper2.pdf")

        claims = analyze_cross_paper_claims([ext1, ext2])
        self.assertIn("pairs", claims)
        self.assertIn("disclaimer", claims)


if __name__ == "__main__":
    unittest.main()
