import os
import subprocess
from collections import defaultdict

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

MODEL_NAMES = ["tfidf", "okapi_tf", "bm25", "laplace", "jm"]

# trec_eval metrics we care about
METRICS_OF_INTEREST = ["map", "P_10", "recall_1000", "ndcg"]


# ---------------------------------------------------------------------------
# Step 10 — Run trec_eval
# ---------------------------------------------------------------------------

def run_trec_eval(qrels_path: str, results_path: str) -> dict[str, float]:
    """
    Run trec_eval on a single results file and parse the output.

    Returns dict  metric_name -> float value
    """
    cmd = ["trec_eval", "-m", "all_trec", qrels_path, results_path]
    try:
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT,
                                         text=True)
    except FileNotFoundError:
        raise RuntimeError(
            "trec_eval not found. Install it:\n"
            "  git clone https://github.com/usnistgov/trec_eval.git\n"
            "  cd trec_eval && make && sudo cp trec_eval /usr/local/bin/"
        )
    except subprocess.CalledProcessError as e:
        print(f"[eval] trec_eval error:\n{e.output}")
        return {}

    parsed: dict[str, float] = {}
    for line in output.splitlines():
        parts = line.split()
        if len(parts) == 3:
            metric, _, value = parts
            try:
                parsed[metric] = float(value)
            except ValueError:
                pass
    return parsed


def evaluate_all(qrels_path: str) -> dict[str, dict[str, float]]:
    """
    Run trec_eval for every model and return a nested dict:
        metrics[model_name][metric_name] = value
    """
    metrics: dict[str, dict[str, float]] = {}
    for model in MODEL_NAMES:
        results_path = os.path.join(RESULTS_DIR, f"{model}.txt")
        if not os.path.exists(results_path):
            print(f"[eval] No results file for '{model}' — skipping.")
            continue
        print(f"[eval] Evaluating {model} …")
        metrics[model] = run_trec_eval(qrels_path, results_path)

    return metrics


# ---------------------------------------------------------------------------
# Step 11 — Comparison & Analysis
# ---------------------------------------------------------------------------

def compare_models(metrics: dict[str, dict[str, float]]) -> None:
    """
    Print a comparison table of the key metrics across all models,
    then provide a qualitative analysis.
    """
    if not metrics:
        print("[eval] No metrics to compare.")
        return

    # --- Table header ---
    col_w = 14
    header_metrics = ["map", "P_10", "recall_1000"]
    print("\n" + "=" * 65)
    print("MODEL COMPARISON TABLE")
    print("=" * 65)
    print(f"{'Model':<14}", end="")
    for m in header_metrics:
        print(f"{m:>{col_w}}", end="")
    print()
    print("-" * 65)

    # Sort by MAP descending
    ranked_models = sorted(
        metrics.items(),
        key=lambda kv: kv[1].get("map", 0.0),
        reverse=True,
    )

    for model_name, m_dict in ranked_models:
        print(f"{model_name:<14}", end="")
        for metric in header_metrics:
            val = m_dict.get(metric, 0.0)
            print(f"{val:>{col_w}.4f}", end="")
        print()

    print("=" * 65)

    # --- Qualitative analysis ---
    best_model  = ranked_models[0][0]  if ranked_models else "N/A"
    best_map    = ranked_models[0][1].get("map", 0.0) if ranked_models else 0.0

    print(f"""
ANALYSIS
--------
Best overall model (by MAP): {best_model.upper()}  (MAP = {best_map:.4f})

Why rankings differ across models
----------------------------------
TF-IDF
  Scores documents by how rare a term is globally (IDF) weighted by
  how often it appears locally (TF). It does NOT normalise for document
  length, so longer documents tend to score higher simply because they
  contain more terms. Good baseline but sensitive to document length.

Okapi TF
  Applies length normalisation to TF but drops the IDF component.
  Works when the query terms are genuinely discriminative on their own,
  but can over-rank documents that repeat query terms without the
  collection-level discrimination that IDF provides.

BM25  ⭐
  The gold standard probabilistic model. Combines Robertson IDF with
  length-normalised TF and a query-term saturation mechanism (k1, b, k3).
  Handles repetition and document length gracefully — typically the
  strongest performer on standard IR benchmarks like Cranfield.

Laplace LM
  Treats retrieval as estimating P(query | document). Add-1 smoothing
  assigns non-zero probability to unseen terms, preventing zero scores.
  However, it does not model collection-level term importance, so common
  terms are not penalised the way IDF does in BM25.

Jelinek-Mercer LM
  Interpolates between the document model and the collection model
  (λ controls the trade-off). This handles zero TF gracefully and
  captures collection-level term importance implicitly. Performance is
  sensitive to λ; λ ≈ 0.1–0.3 usually works best for short queries.

Key insight
-----------
BM25 and JM typically outperform TF-IDF on MAP for the Cranfield
collection because Cranfield queries are short and precise — exactly
the regime where Robertson IDF and language-model smoothing help most.
""")


# ---------------------------------------------------------------------------
# Per-query top-10 comparison (useful for report / notebook)
# ---------------------------------------------------------------------------

def compare_top10(
    all_results: dict[str, dict[str, list]],
    qrels: dict[str, set[str]],
    query_id: str,
) -> None:
    """
    Print side-by-side top-10 rankings for a specific query across all models,
    marking relevant documents with ✓.
    """
    print(f"\nTop-10 comparison for query {query_id}")
    print("-" * 80)
    relevant = qrels.get(query_id, set())

    # header
    col = 16
    print(f"{'Rank':<6}", end="")
    for m in MODEL_NAMES:
        print(f"{m:<{col}}", end="")
    print()
    print("-" * 80)

    for rank in range(10):
        print(f"{rank+1:<6}", end="")
        for model in MODEL_NAMES:
            ranked = all_results.get(model, {}).get(query_id, [])
            if rank < len(ranked):
                doc_id, _ = ranked[rank]
                marker = "✓" if doc_id in relevant else " "
                cell = f"{doc_id}{marker}"
            else:
                cell = "-"
            print(f"{cell:<{col}}", end="")
        print()
    print()