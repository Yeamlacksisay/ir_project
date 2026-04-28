"""
ranker.py
---------
Steps 8 & 9 — Score all documents for each query and write ranked results.

All scoring is done in pure Python from in-memory caches — no ES calls
during ranking. helper.warm() loads everything upfront in one bulk request.
"""

import os
from tqdm import tqdm

from helpers import ESHelper
from models import (
    score_tfidf,
    score_okapi_tf,
    score_bm25,
    score_laplace,
    score_jelinek_mercer,
)

MODELS = {
    "tfidf":    score_tfidf,
    "okapi_tf": score_okapi_tf,
    "bm25":     score_bm25,
    "laplace":  score_laplace,
    "jm":       score_jelinek_mercer,
}

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
TOP_K       = 100


def tokenise(text: str) -> list[str]:
    return text.split()


def rank_documents(
    query_terms: list[str],
    doc_ids: list[str],
    score_fn,
    helper: ESHelper,
) -> list[tuple[str, float]]:
    scores = [(doc_id, score_fn(query_terms, doc_id, helper)) for doc_id in doc_ids]
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:TOP_K]


def run_all_models(
    queries: list[dict],
    helper: ESHelper,
    models: dict | None = None,
) -> dict[str, dict[str, list[tuple[str, float]]]]:

    if models is None:
        models = MODELS

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load ALL term vectors in one bulk ES request — takes ~10s, then fast
    helper.warm()

    all_doc_ids = helper.get_all_doc_ids()
    print(f"[ranker] {len(all_doc_ids)} documents, {len(queries)} queries, {len(models)} models")

    all_results: dict[str, dict[str, list]] = {m: {} for m in models}

    for model_name, score_fn in models.items():
        print(f"\n[ranker] === Model: {model_name.upper()} ===")
        out_path = os.path.join(RESULTS_DIR, f"{model_name}.txt")

        with open(out_path, "w", encoding="utf-8") as fout:
            for query in tqdm(queries, desc=model_name):
                qid   = query["qid"]
                terms = tokenise(query["text"])
                if not terms:
                    continue

                ranked = rank_documents(terms, all_doc_ids, score_fn, helper)
                all_results[model_name][qid] = ranked

                for rank, (doc_id, score) in enumerate(ranked, start=1):
                    fout.write(f"{qid} Q0 {doc_id} {rank} {score:.6f} {model_name}\n")

        print(f"[ranker] Results written → {out_path}")

    return all_results