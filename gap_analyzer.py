from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re


def discover_research_gaps(papers: list) -> dict:
    """
    Cluster and group author-stated limitations and future-work directions across papers.
    Uses TF-IDF representation and cosine similarity clustering.
    Maintains explicit provenance (paper name and page number) for all statements.
    """
    raw_statements = []

    for paper in papers:
        p_name = paper.get("filename", "Paper")
        dims = paper.get("dimensions", {})

        # 1. Limitations
        if "Limitations" in dims and dims["Limitations"]["evidence"]:
            for item in dims["Limitations"]["evidence"]:
                raw_statements.append({
                    "paper_name": p_name,
                    "page_number": item.get("page_number", 1),
                    "text": item.get("text", "").strip(),
                    "type": "Explicit Author-Stated Limitation"
                })

        # 2. Future Work
        if "Future Work" in dims and dims["Future Work"]["evidence"]:
            for item in dims["Future Work"]["evidence"]:
                raw_statements.append({
                    "paper_name": p_name,
                    "page_number": item.get("page_number", 1),
                    "text": item.get("text", "").strip(),
                    "type": "Explicit Future Direction"
                })

    # Deduplicate statements
    seen = set()
    unique_statements = []
    for s in raw_statements:
        key = (s["paper_name"], s["text"][:50].lower())
        if key not in seen and len(s["text"]) >= 25:
            seen.add(key)
            unique_statements.append(s)

    if not unique_statements:
        return {
            "candidate_gaps": [],
            "explicit_statements": [],
            "disclaimer": "No explicit limitation or future-work statements were detected via linguistic cues in the uploaded documents."
        }

    if len(unique_statements) < 2:
        return {
            "candidate_gaps": [{
                "theme": "Single Document Limitation / Open Direction",
                "is_inferred": False,
                "statements": unique_statements,
                "paper_count": 1,
                "summary": unique_statements[0]["text"]
            }],
            "explicit_statements": unique_statements,
            "disclaimer": "Analysis is based on 1 available statement. Multi-paper cross-gap synthesis requires multiple papers with limitation disclosures."
        }

    # Vectorize statements for clustering
    texts = [s["text"] for s in unique_statements]
    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=2500,
            ngram_range=(1, 2)
        )
        tfidf_mat = vectorizer.fit_transform(texts)
        sim_mat = cosine_similarity(tfidf_mat)
    except Exception:
        sim_mat = np.eye(len(unique_statements))

    # Transparent agglomerative grouping via thresholding (threshold = 0.20)
    assigned = [False] * len(unique_statements)
    clusters = []

    for i in range(len(unique_statements)):
        if assigned[i]:
            continue
        cluster_members = [unique_statements[i]]
        assigned[i] = True

        for j in range(i + 1, len(unique_statements)):
            if not assigned[j] and sim_mat[i, j] >= 0.20:
                cluster_members.append(unique_statements[j])
                assigned[j] = True

        papers_in_cluster = list({m["paper_name"] for m in cluster_members})
        is_cross_paper = len(papers_in_cluster) > 1

        # Derive a concise theme label from the most central statement
        theme_lead = cluster_members[0]["text"]
        # Truncate theme lead gracefully
        theme_title = theme_lead[:85] + "..." if len(theme_lead) > 85 else theme_lead

        clusters.append({
            "theme": theme_title,
            "is_inferred": is_cross_paper,
            "gap_category": "Cross-Paper Convergent Limitation" if is_cross_paper else "Document-Specific Constraint",
            "statements": cluster_members,
            "papers": papers_in_cluster,
            "paper_count": len(papers_in_cluster),
            "evidence_count": len(cluster_members)
        })

    # Sort clusters by cross-paper frequency first, then evidence count
    clusters.sort(key=lambda c: (-c["paper_count"], -c["evidence_count"]))

    return {
        "candidate_gaps": clusters,
        "explicit_statements": unique_statements,
        "disclaimer": (
            "Scope Notice: Identified research gaps reflect only constraints explicitly stated within "
            "the uploaded PDF collection. They do not constitute proof of global novelty or exhaustiveness "
            "across the wider scientific literature."
        )
    }


CONCORDANCE_CUES = [
    "outperformed", "improved", "superior", "consistent", "effective",
    "achieves", "higher accuracy", "confirms", "demonstrates effectiveness",
    "robust", "state-of-the-art", "significant improvement"
]

DIVERGENCE_CUES = [
    "however", "fails", "degrades", "inferior", "limitation", "trade-off",
    "overhead", "unlike", "contrast", "vulnerable", "underperforms",
    "bottleneck", "constrained by", "computational cost"
]


def analyze_cross_paper_claims(papers: list) -> dict:
    """
    Identify related claims across different research papers using TF-IDF semantic similarity
    and heuristic stance detection.
    Never asserts definitive logical contradiction without human verification.
    """
    if len(papers) < 2:
        return {
            "pairs": [],
            "disclaimer": "At least two research papers are required to analyze cross-paper claim alignment."
        }

    claims_by_paper = []
    for paper in papers:
        p_name = paper.get("filename", "Paper")
        dims = paper.get("dimensions", {})

        # Collect claims from Findings and Methodology
        for category in ["Findings", "Methodology"]:
            if category in dims and dims[category]["evidence"]:
                for item in dims[category]["evidence"]:
                    t = item.get("text", "").strip()
                    if len(t) >= 30:
                        claims_by_paper.append({
                            "paper_name": p_name,
                            "page_number": item.get("page_number", 1),
                            "category": category,
                            "text": t
                        })

    if len(claims_by_paper) < 2:
        return {
            "pairs": [],
            "disclaimer": "Insufficient extracted claims across papers to evaluate alignment."
        }

    texts = [c["text"] for c in claims_by_paper]
    try:
        vectorizer = TfidfVectorizer(stop_words="english", max_features=3000)
        tfidf_mat = vectorizer.fit_transform(texts)
        sim_mat = cosine_similarity(tfidf_mat)
    except Exception:
        return {"pairs": [], "disclaimer": "Unable to compute TF-IDF matrix for claims."}

    candidate_pairs = []

    for i in range(len(claims_by_paper)):
        for j in range(i + 1, len(claims_by_paper)):
            claim_a = claims_by_paper[i]
            claim_b = claims_by_paper[j]

            # Only compare claims originating from DIFFERENT papers
            if claim_a["paper_name"] == claim_b["paper_name"]:
                continue

            similarity = float(sim_mat[i, j])

            # Filter for semantic topic relatedness
            if similarity >= 0.22:
                text_a_lower = claim_a["text"].lower()
                text_b_lower = claim_b["text"].lower()

                has_concordance_a = any(cue in text_a_lower for cue in CONCORDANCE_CUES)
                has_concordance_b = any(cue in text_b_lower for cue in CONCORDANCE_CUES)
                has_divergence_a = any(cue in text_a_lower for cue in DIVERGENCE_CUES)
                has_divergence_b = any(cue in text_b_lower for cue in DIVERGENCE_CUES)

                # Heuristic stance classification
                if has_divergence_a or has_divergence_b:
                    label = "Potential Divergence / Differing Perspective Candidate"
                    badge_color = "red"
                    rationale = "Contains contrastive or limitation linguistic markers in one or both claims."
                elif has_concordance_a and has_concordance_b:
                    label = "Potential Agreement / Concordance Candidate"
                    badge_color = "green"
                    rationale = "Both claims report performance gains or positive validation."
                else:
                    label = "Related Cross-Paper Claims (Shared Topic)"
                    badge_color = "blue"
                    rationale = "High lexical/semantic overlap on shared methodology or evaluation dimension."

                candidate_pairs.append({
                    "paper_a": claim_a["paper_name"],
                    "page_a": claim_a["page_number"],
                    "text_a": claim_a["text"],
                    "paper_b": claim_b["paper_name"],
                    "page_b": claim_b["page_number"],
                    "text_b": claim_b["text"],
                    "similarity": round(similarity, 3),
                    "label": label,
                    "badge_color": badge_color,
                    "rationale": rationale,
                    "status": "Candidate requiring human review"
                })

    # Sort candidate pairs by similarity descending
    candidate_pairs.sort(key=lambda p: -p["similarity"])

    return {
        "pairs": candidate_pairs[:20],
        "disclaimer": (
            "Methodology Disclaimer: Claim alignment is evaluated using TF-IDF lexical-semantic overlap and "
            "stance marker heuristics. Stance classifications are candidate hypotheses intended to assist literature "
            "review and MUST be verified by a human researcher. Statistical similarity does not equal semantic "
            "entailment or logical contradiction."
        )
    }
