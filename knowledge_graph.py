
import spacy
import networkx as nx
from collections import Counter
from itertools import combinations

# Load the English NLP model
nlp = spacy.load("en_core_web_sm")


def build_knowledge_graph(text, max_concepts=30):
    """
    Extract concepts from text and connect concepts
    that appear together in the same sentence.
    Returns a NetworkX graph.
    """

    doc = nlp(text)
    graph = nx.Graph()

    # Extract named entities and meaningful noun phrases
    concepts = []

    for ent in doc.ents:
        phrase = ent.text.strip()
        if len(phrase) > 2:
            concepts.append(phrase)

    for chunk in doc.noun_chunks:
        phrase = chunk.text.strip()
        if len(phrase) > 2:
            concepts.append(phrase)

    # Normalize repeated concepts
    counts = Counter(
        phrase.lower() for phrase in concepts
    )

    # Keep the most frequent concepts
    selected = counts.most_common(max_concepts)

    # Use lowercase names to merge duplicates
    for concept, frequency in selected:
        graph.add_node(
            concept,
            frequency=frequency
        )

    allowed = set(graph.nodes)

    # Connect concepts appearing in the same sentence
    for sentence in doc.sents:
        sentence_concepts = set()

        for ent in sentence.ents:
            phrase = ent.text.strip().lower()
            if phrase in allowed:
                sentence_concepts.add(phrase)

        for chunk in sentence.noun_chunks:
            phrase = chunk.text.strip().lower()
            if phrase in allowed:
                sentence_concepts.add(phrase)

        for first, second in combinations(
            sorted(sentence_concepts), 2
        ):
            if first != second:
                if graph.has_edge(first, second):
                    graph[first][second]["weight"] += 1
                else:
                    graph.add_edge(
                        first,
                        second,
                        weight=1
                    )

    return graph


def get_graph_data(graph):
    """
    Convert the graph into node and edge lists
    that can be displayed in Streamlit.
    """

    nodes = [
        {
            "id": node,
            "label": node,
            "frequency": data.get("frequency", 1)
        }
        for node, data in graph.nodes(data=True)
    ]

    edges = [
        {
            "source": first,
            "target": second,
            "weight": data.get("weight", 1)
        }
        for first, second, data in graph.edges(data=True)
    ]

    return nodes, edges
