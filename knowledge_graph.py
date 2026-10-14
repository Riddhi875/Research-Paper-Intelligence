
import spacy
import networkx as nx
from collections import Counter, defaultdict
from itertools import combinations

# Load the English NLP model once
nlp = spacy.load("en_core_web_sm")


def build_knowledge_graph(text, max_concepts=12):
    """
    Build a smaller, easier-to-read knowledge graph.

    Nodes are meaningful phrases extracted from the paper.
    Edges show that two concepts appeared in the same sentence.
    An edge does NOT necessarily mean one concept causes another.
    """

    doc = nlp(text)
    graph = nx.Graph()

    concept_counts = Counter()
    concept_sentences = defaultdict(list)

    # Extract meaningful noun phrases and named entities
    for sentence in doc.sents:
        sentence_concepts = set()

        for chunk in sentence.noun_chunks:
            phrase = chunk.text.strip().lower()
            phrase = " ".join(phrase.split())

            # Ignore very short or uninformative phrases
            if len(phrase) < 4:
                continue

            if len(phrase.split()) > 6:
                continue

            if not any(token.is_alpha for token in nlp(phrase)):
                continue

            sentence_concepts.add(phrase)

        for ent in sentence.ents:
            phrase = " ".join(ent.text.strip().lower().split())

            if len(phrase) >= 4:
                sentence_concepts.add(phrase)

        for phrase in sentence_concepts:
            concept_counts[phrase] += 1

            original_sentence = sentence.text.strip()

            if (
                original_sentence
                and original_sentence not in concept_sentences[phrase]
            ):
                concept_sentences[phrase].append(original_sentence)

    # Select the most frequent phrases
    selected = concept_counts.most_common(max_concepts)
    allowed = {concept for concept, _ in selected}

    for concept, frequency in selected:
        graph.add_node(
            concept,
            frequency=frequency,
            evidence=concept_sentences[concept][:3]
        )

    # Connect concepts that appear together in a sentence
    for sentence in doc.sents:
        sentence_concepts = set()

        for chunk in sentence.noun_chunks:
            phrase = " ".join(chunk.text.strip().lower().split())

            if phrase in allowed:
                sentence_concepts.add(phrase)

        for ent in sentence.ents:
            phrase = " ".join(ent.text.strip().lower().split())

            if phrase in allowed:
                sentence_concepts.add(phrase)

        for first, second in combinations(sorted(sentence_concepts), 2):
            if first == second:
                continue

            if graph.has_edge(first, second):
                graph[first][second]["weight"] += 1
            else:
                graph.add_edge(first, second, weight=1)

    # Keep only stronger connections to reduce visual clutter
    edges_to_remove = [
        (first, second)
        for first, second, data in graph.edges(data=True)
        if data.get("weight", 1) < 2
    ]

    graph.remove_edges_from(edges_to_remove)

    return graph


def get_graph_data(graph):
    """Convert graph nodes and edges into simple Python lists."""

    nodes = [
        {
            "id": node,
            "label": node,
            "frequency": data.get("frequency", 1),
            "evidence": data.get("evidence", [])
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


def get_concept_details(graph, concept):
    """Return supporting sentences and related concepts."""

    if concept not in graph:
        return None

    related = sorted(
        graph[concept].items(),
        key=lambda item: item[1].get("weight", 1),
        reverse=True
    )

    return {
        "concept": concept,
        "frequency": graph.nodes[concept].get("frequency", 1),
        "evidence": graph.nodes[concept].get("evidence", []),
        "related": [
            {
                "concept": name,
                "co_occurrences": data.get("weight", 1)
            }
            for name, data in related
        ]
    }
