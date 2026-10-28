from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class MultiPaperSearchEngine:
    """
    TF-IDF based search engine for cross-paper literature retrieval.
    Retains source paper and page provenance for all retrieved passages.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=15000,
            sublinear_tf=True
        )
        self.passages = []  # list of {"text": str, "paper_name": str, "page_number": int}
        self.matrix = None

    def index_papers(self, papers: list) -> int:
        """
        Index multiple papers.
        Each item in `papers` is expected to have:
        - "filename": str
        - "sentences": list of {"text": str, "page_number": int, "paper_name": str}
          or "pages": list of {"page_number": int, "text": str}
        """
        self.passages = []

        for paper in papers:
            filename = paper.get("filename", "Unknown Paper")
            # If pre-segmented sentences with provenance exist, use them
            if "tagged_sentences" in paper and paper["tagged_sentences"]:
                for s in paper["tagged_sentences"]:
                    t = s.get("text", "").strip()
                    if len(t) >= 20:
                        self.passages.append({
                            "text": t,
                            "paper_name": s.get("paper_name", filename),
                            "page_number": s.get("page_number", 1)
                        })
            elif "pages" in paper:
                from processor import split_sentences
                for page in paper["pages"]:
                    p_num = page.get("page_number", 1)
                    sents = split_sentences(page.get("text", ""))
                    for s in sents:
                        self.passages.append({
                            "text": s,
                            "paper_name": filename,
                            "page_number": p_num
                        })

        if not self.passages:
            self.matrix = None
            return 0

        texts = [p["text"] for p in self.passages]
        self.matrix = self.vectorizer.fit_transform(texts)
        return len(self.passages)

    def search(self, query: str, top_k: int = 5, min_threshold: float = 0.05) -> list:
        """
        Search across all indexed research papers.
        Returns top passages with relevance scores, source paper name, and page provenance.
        Does not return irrelevant passages with near-zero scores.
        """
        if self.matrix is None or not query or not query.strip():
            return []

        clean_query = query.strip()
        try:
            query_vector = self.vectorizer.transform([clean_query])
        except Exception:
            return []

        scores = cosine_similarity(query_vector, self.matrix).flatten()

        # Filter indices by threshold
        ranked_indices = scores.argsort()[::-1]
        results = []

        for idx in ranked_indices:
            score = float(scores[idx])
            if score < min_threshold:
                break  # Further scores will be even lower

            item = self.passages[idx]
            results.append({
                "sentence": item["text"],
                "score": round(score, 4),
                "paper_name": item["paper_name"],
                "page_number": item["page_number"]
            })

            if len(results) >= top_k:
                break

        return results

    def get_stats(self) -> dict:
        """Return index statistics."""
        return {
            "total_passages": len(self.passages),
            "vocabulary_size": len(self.vectorizer.vocabulary_) if hasattr(self.vectorizer, "vocabulary_") and self.vectorizer.vocabulary_ else 0,
        }


# Backward-compatible wrapper for single paper usage
class ResearchPaperSearch:
    def __init__(self):
        self.engine = MultiPaperSearchEngine()
        self.sentences = []
        self.matrix = None

    def index_text(self, text: str):
        from processor import split_sentences
        sents = split_sentences(text)
        self.sentences = sents
        tagged = [{"text": s, "page_number": 1, "paper_name": "Current Paper"} for s in sents]
        self.engine.index_papers([{"filename": "Current Paper", "tagged_sentences": tagged}])
        self.matrix = self.engine.matrix
        return len(sents)

    def search(self, query: str, top_k: int = 5):
        raw_results = self.engine.search(query, top_k=top_k, min_threshold=0.01)
        return [{"sentence": r["sentence"], "score": r["score"]} for r in raw_results]
