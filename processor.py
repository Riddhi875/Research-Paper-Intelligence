
import pymupdf
import re
import spacy

# Load the English NLP model once
nlp = spacy.load("en_core_web_sm")


def extract_text(pdf_bytes):
    """Extract text from an uploaded PDF."""

    document = pymupdf.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    pages = []

    for page in document:
        pages.append(page.get_text("text"))

    document.close()

    full_text = "\n".join(pages)
    full_text = re.sub(r"[ \t]+", " ", full_text)
    full_text = re.sub(r"\n{3,}", "\n\n", full_text)

    return full_text.strip()


def split_sentences(text):
    """Split text into readable sentences."""

    doc = nlp(text)

    return [
        sentence.text.strip()
        for sentence in doc.sents
        if len(sentence.text.strip()) > 30
    ]


def summarize_text(text, max_sentences=5):
    """Create a simple extractive summary."""

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    sentences = split_sentences(text)

    if len(sentences) <= max_sentences:
        return sentences

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000
    )

    matrix = vectorizer.fit_transform(sentences)
    similarity = cosine_similarity(matrix)

    scores = similarity.mean(axis=1)

    best_indices = np.argsort(scores)[-max_sentences:]
    best_indices = sorted(best_indices)

    return [sentences[i] for i in best_indices]


def create_study_notes(text):
    """
    Create structured study notes using text already present
    in the research paper. This function does not invent
    missing sections or generate new scientific claims.
    """

    doc = nlp(text)
    sentences = [
        sentence.text.strip()
        for sentence in doc.sents
        if len(sentence.text.strip()) > 30
    ]

    if not sentences:
        return {
            "Paper overview": [],
            "Problem and motivation": [],
            "Methodology": [],
            "Results and findings": [],
            "Limitations and future work": []
        }

    section_keywords = {
        "Paper overview": [
            "we propose", "this paper", "this study",
            "this research", "we present", "our work",
            "the objective", "the aim"
        ],
        "Problem and motivation": [
            "problem", "challenge", "limitation",
            "however", "difficult", "issue", "gap",
            "existing methods", "drawback", "need for"
        ],
        "Methodology": [
            "method", "approach", "model", "algorithm",
            "framework", "dataset", "we use", "we used",
            "proposed system", "experiment", "architecture"
        ],
        "Results and findings": [
            "results", "findings", "accuracy", "performance",
            "outperformed", "improved", "evaluation",
            "achieved", "compared with", "demonstrate"
        ],
        "Limitations and future work": [
            "limitation", "future work", "further research",
            "in the future", "remains a challenge",
            "shortcoming", "we plan", "could be improved"
        ]
    }

    lower_sentences = [s.lower() for s in sentences]
    notes = {}

    used_indices = set()

    for section, keywords in section_keywords.items():
        scored = []

        for index, sentence in enumerate(lower_sentences):
            score = sum(
                1 for keyword in keywords if keyword in sentence
            )

            if score > 0:
                scored.append((score, index))

        # Highest keyword match first, preserving source sentences
        scored.sort(key=lambda item: (-item[0], item[1]))

        selected_indices = []

        for score, index in scored:
            if index not in selected_indices:
                selected_indices.append(index)

            if len(selected_indices) >= 5:
                break

        selected_indices.sort()
        notes[section] = [sentences[i] for i in selected_indices]
        used_indices.update(selected_indices)

    # Provide a fallback if the paper lacks recognizable section cues
    if not any(notes.values()):
        notes["Paper overview"] = sentences[:8]

    return notes
