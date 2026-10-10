
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
        page_text = page.get_text("text")
        pages.append(page_text)

    document.close()

    full_text = "\n".join(pages)

    # Clean excessive spaces
    full_text = re.sub(r"[ \t]+", " ", full_text)
    full_text = re.sub(r"\n{3,}", "\n\n", full_text)

    return full_text.strip()


def split_sentences(text):
    """Split a paper into sentences."""

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
