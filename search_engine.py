
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ResearchPaperSearch:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=10000
        )
        self.sentences = []
        self.matrix = None

    def index_text(self, text):
        """
        Split the paper into sentences and create
        a TF-IDF index for searching.
        """

        import spacy

        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)

        self.sentences = [
            sentence.text.strip()
            for sentence in doc.sents
            if len(sentence.text.strip()) > 20
        ]

        if not self.sentences:
            self.matrix = None
            return 0

        self.matrix = self.vectorizer.fit_transform(
            self.sentences
        )

        return len(self.sentences)

    def search(self, query, top_k=5):
        """
        Return the most relevant sentences for a query.
        """

        if self.matrix is None or not query.strip():
            return []

        query_vector = self.vectorizer.transform([query])

        scores = cosine_similarity(
            query_vector,
            self.matrix
        ).flatten()

        # Highest relevance scores first
        indices = scores.argsort()[::-1]

        results = []

        for index in indices:
            if scores[index] <= 0:
                continue

            results.append({
                "sentence": self.sentences[index],
                "score": float(scores[index])
            })

            if len(results) >= top_k:
                break

        return results
