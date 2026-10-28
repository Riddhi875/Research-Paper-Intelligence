import re
import pymupdf
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load spaCy NLP model safely with graceful fallback
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = None


def clean_text(text: str) -> str:
    """Normalize whitespace, handle linebreaks, and remove non-printable characters."""
    if not text:
        return ""
    # Replace carriage returns
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Fix hyphenated line breaks (e.g., "algo-\nrithm" -> "algorithm")
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
    # Collapse irregular horizontal whitespace
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse excessive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_text(pdf_bytes: bytes) -> str:
    """Extract raw combined text from an uploaded PDF (backward compatible)."""
    result = extract_pdf_pages(pdf_bytes)
    return result["full_text"]


def extract_pdf_pages(pdf_bytes: bytes, filename: str = "document.pdf") -> dict:
    """
    Extract text page-by-page from an uploaded PDF.
    Preserves page provenance and diagnoses scanned or unreadable documents.
    """
    try:
        document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        return {
            "filename": filename,
            "pages": [],
            "full_text": "",
            "total_pages": 0,
            "status": "Corrupted or Unreadable PDF",
            "error_message": str(exc),
        }

    pages = []
    total_chars = 0

    for idx, page in enumerate(document, start=1):
        raw_text = page.get_text("text") or ""
        cleaned = clean_text(raw_text)
        char_count = len(cleaned)
        total_chars += char_count
        pages.append({
            "page_number": idx,
            "text": cleaned,
            "char_count": char_count,
        })

    document.close()

    full_text = "\n\n".join(p["text"] for p in pages if p["text"])

    if len(pages) == 0:
        status = "Empty Document"
    elif total_chars < 50:
        status = "Scanned / OCR Needed (Empty Text Layer)"
    else:
        status = "Success"

    return {
        "filename": filename,
        "pages": pages,
        "full_text": full_text.strip(),
        "total_pages": len(pages),
        "status": status,
        "error_message": None,
    }


def split_sentences(text: str, min_chars: int = 25) -> list:
    """Split text into readable sentences using spaCy, with regex fallback."""
    if not text or not text.strip():
        return []

    if nlp is not None:
        try:
            # Process in safe chunks to avoid spaCy max_length overflow
            doc = nlp(text[:200000])
            sentences = [
                s.text.strip()
                for s in doc.sents
                if len(s.text.strip()) >= min_chars
            ]
            if sentences:
                return sentences
        except Exception:
            pass

    # Regex fallback sentence segmentation
    raw_sents = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", text)
    return [
        s.strip()
        for s in raw_sents
        if len(s.strip()) >= min_chars
    ]


def split_sentences_with_provenance(pages: list, filename: str, min_chars: int = 25) -> list:
    """
    Split all pages into sentences, tagging each sentence with its source paper and page number.
    Returns list of dicts: [{"paper_name": str, "page_number": int, "text": str}]
    """
    provenance_sentences = []
    for page in pages:
        p_num = page.get("page_number", 1)
        p_text = page.get("text", "")
        if not p_text:
            continue
        sents = split_sentences(p_text, min_chars=min_chars)
        for s in sents:
            provenance_sentences.append({
                "paper_name": filename,
                "page_number": p_num,
                "text": s,
            })
    return provenance_sentences


def summarize_text(text: str, max_sentences: int = 5) -> list:
    """Create a simple extractive summary using TF-IDF sentence centrality."""
    sentences = split_sentences(text)
    if len(sentences) <= max_sentences:
        return sentences

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=5000,
            sublinear_tf=True
        )
        matrix = vectorizer.fit_transform(sentences)
        similarity = cosine_similarity(matrix)
        scores = similarity.mean(axis=1)
        best_indices = np.argsort(scores)[-max_sentences:]
        best_indices = sorted(best_indices)
        return [sentences[i] for i in best_indices]
    except Exception:
        return sentences[:max_sentences]


# Section and linguistic cue definitions for paper dimension extraction
DIMENSION_PATTERNS = {
    "Problem": [
        "we address the problem", "the main challenge", "limitation of existing",
        "however, existing", "a major bottleneck", "remains difficult",
        "the objective of this", "aims to resolve", "lack of", "inability to",
        "fails to", "traditional approaches suffer", "gap in", "the motivation"
    ],
    "Methodology": [
        "we propose", "we introduce", "our method", "our approach",
        "in this paper, we develop", "we design an architecture", "the proposed framework",
        "the model consists of", "we employ a novel", "pipeline includes",
        "algorithm", "we train", "we implement", "transformer-based", "neural network"
    ],
    "Datasets": [
        "we evaluate on", "benchmarks", "dataset", "datasets", "corpus",
        "experimental setup", "tested on the", "collected from", "annotated",
        "imagenet", "glue", "squad", "cifar", "mnist", "trec", "real-world data"
    ],
    "Findings": [
        "outperforms", "achieves state-of-the-art", "improved performance",
        "results demonstrate", "experimental results show", "accuracy of",
        "f1 score", "bleu score", "statistically significant", "consistently outperforms",
        "reduces error", "higher precision", "superior to baseline"
    ],
    "Limitations": [
        "limitation", "limitations", "drawback", "shortcoming",
        "our approach fails when", "requires high computational", "restricted to",
        "does not generalize", "trade-off", "performance degrades",
        "assumes that", "constrained by", "computational overhead"
    ],
    "Future Work": [
        "future work", "in the future", "further research is needed",
        "we plan to", "can be extended to", "remains an open question",
        "future directions include", "future investigations should",
        "promising avenue for", "could be improved by"
    ]
}


def extract_paper_dimensions(pages_data: list, filename: str) -> dict:
    """
    Extract candidate research dimensions:
    Problem, Methodology, Datasets, Findings, Limitations, and Future Work.
    Preserves exact evidence sentences and page numbers.
    Never presents a missing field as proof that the paper does not address it.
    """
    tagged_sentences = split_sentences_with_provenance(pages_data, filename, min_chars=28)

    dimensions = {
        "Problem": {"summary": "Not explicitly found in extracted text", "evidence": [], "status": "Not detected"},
        "Methodology": {"summary": "Not explicitly found in extracted text", "evidence": [], "status": "Not detected"},
        "Datasets": {"summary": "Not explicitly found in extracted text", "evidence": [], "status": "Not detected"},
        "Findings": {"summary": "Not explicitly found in extracted text", "evidence": [], "status": "Not detected"},
        "Limitations": {"summary": "Not explicitly found in extracted text", "evidence": [], "status": "Not detected"},
        "Future Work": {"summary": "Not explicitly found in extracted text", "evidence": [], "status": "Not detected"},
    }

    if not tagged_sentences:
        return dimensions

    for dim_name, cues in DIMENSION_PATTERNS.items():
        matched = []
        for item in tagged_sentences:
            lower_text = item["text"].lower()
            # Multi-word cues receive higher weight than single words
            score = sum(len(cue.split()) * 2 for cue in cues if cue in lower_text)
            
            # Penalize limitation score if the sentence is primarily reporting positive results
            if dim_name == "Limitations":
                if any(pos in lower_text for pos in ["accuracy of", "consistently outperforming", "outperforms baseline"]):
                    score -= 4

            if score > 0:
                matched.append({
                    "score": score,
                    "text": item["text"],
                    "page_number": item["page_number"],
                })

        # Sort matches by weighted score descending, then brevity for readability
        matched.sort(key=lambda m: (-m["score"], len(m["text"])))

        if matched:
            top_evidence = [
                {"text": m["text"], "page_number": m["page_number"]}
                for m in matched[:4]
            ]
            dimensions[dim_name] = {
                "summary": top_evidence[0]["text"],
                "evidence": top_evidence,
                "status": "Found with evidence",
            }
        else:
            dimensions[dim_name] = {
                "summary": "Not explicitly detected via linguistic patterns (requires manual review)",
                "evidence": [],
                "status": "Uncertain / Not detected",
            }

    return dimensions


def extract_key_concepts(text: str, max_concepts: int = 15) -> list:
    """Extract frequent noun phrases and named entities from text."""
    if not text:
        return []

    concepts = Counter()

    if nlp is not None:
        try:
            doc = nlp(text[:150000])
            for chunk in doc.noun_chunks:
                phrase = " ".join(chunk.text.lower().split())
                if 4 <= len(phrase) <= 40 and len(phrase.split()) <= 4:
                    if not any(token.is_stop for token in nlp(phrase)):
                        concepts[phrase] += 1
            for ent in doc.ents:
                phrase = " ".join(ent.text.lower().split())
                if 3 <= len(phrase) <= 40:
                    concepts[phrase] += 2
        except Exception:
            pass

    if not concepts:
        # Fallback to alphanumeric key phrases
        words = re.findall(r"\b[A-Za-z][A-Za-z0-9\-]{3,}\b", text.lower())
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        for w in words:
            if w not in ENGLISH_STOP_WORDS:
                concepts[w] += 1

    return [concept for concept, _ in concepts.most_common(max_concepts)]


def create_study_notes(text: str) -> dict:
    """Backward-compatible study notes generation."""
    sentences = split_sentences(text)
    if not sentences:
        return {
            "Paper overview": [],
            "Problem and motivation": [],
            "Methodology": [],
            "Results and findings": [],
            "Limitations and future work": [],
        }

    section_keywords = {
        "Paper overview": ["we propose", "this paper", "this study", "this research", "we present", "our work"],
        "Problem and motivation": ["problem", "challenge", "limitation", "however", "difficult", "issue", "gap"],
        "Methodology": ["method", "approach", "model", "algorithm", "framework", "dataset", "we use", "architecture"],
        "Results and findings": ["results", "findings", "accuracy", "performance", "outperformed", "improved", "evaluation"],
        "Limitations and future work": ["limitation", "future work", "further research", "in the future", "shortcoming"],
    }

    lower_sentences = [s.lower() for s in sentences]
    notes = {}

    for section, keywords in section_keywords.items():
        scored = []
        for idx, sent in enumerate(lower_sentences):
            score = sum(1 for kw in keywords if kw in sent)
            if score > 0:
                scored.append((score, idx))

        scored.sort(key=lambda item: (-item[0], item[1]))
        selected_indices = [idx for _, idx in scored[:5]]
        selected_indices.sort()
        notes[section] = [sentences[i] for i in selected_indices]

    if not any(notes.values()):
        notes["Paper overview"] = sentences[:6]

    return notes
