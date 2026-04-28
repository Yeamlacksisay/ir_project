"""
main.py
-------
Master orchestrator — runs all 11 steps in order.

Usage
-----
1. Start Elasticsearch:
       docker compose up -d

2. Install dependencies:
       pip install -r requirements.txt

3. Run the full pipeline:
       python main.py

4. (Optional) Force re-index if you change the data:
       python main.py --reindex

Flags
-----
--reindex   : Delete and recreate the ES index before indexing
--skip-index: Skip indexing (useful if index already populated)
--eval-only : Jump straight to evaluation (index must exist)
--query     : Show top-10 comparison for a specific query ID e.g. --query 1
"""

import argparse
import os
import sys

# Make src/ importable regardless of working directory
SRC_DIR = os.path.join(os.path.dirname(__file__), "src")
sys.path.insert(0, SRC_DIR)

from parser             import parse_documents, parse_queries, parse_qrels
from elasticsearch_setup import get_client, create_index, index_documents
from helpers            import ESHelper
from ranker             import run_all_models
from evaluator          import evaluate_all, compare_models, compare_top10


# ---------------------------------------------------------------------------
# Paths — adjust if your files live elsewhere
# ---------------------------------------------------------------------------

DATA_DIR      = os.path.join(os.path.dirname(__file__), "data")
DOCS_XML      = os.path.join(DATA_DIR, "cranfield_docs.xml")
QUERIES_XML   = os.path.join(DATA_DIR, "queries.xml")
QRELS_TXT     = os.path.join(DATA_DIR, "qrels.txt")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Cranfield IR pipeline")
    parser.add_argument("--reindex",    action="store_true",
                        help="Delete and recreate the ES index")
    parser.add_argument("--skip-index", action="store_true",
                        help="Skip document indexing")
    parser.add_argument("--eval-only",  action="store_true",
                        help="Only run evaluation (index must already exist)")
    parser.add_argument("--query",      type=str, default=None,
                        help="Show top-10 per-model comparison for a query ID")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Step 1 — Parse documents
    # ------------------------------------------------------------------
    if not args.eval_only:
        print("\n[STEP 1] Parsing documents …")
        documents = parse_documents(DOCS_XML)

    # ------------------------------------------------------------------
    # Step 2 & 3 — Create index & index documents
    # ------------------------------------------------------------------
    print("\n[STEP 2/3] Connecting to Elasticsearch …")
    client = get_client()

    if not args.eval_only and not args.skip_index:
        create_index(client, force_recreate=args.reindex)
        print("\n[STEP 3] Indexing documents …")
        index_documents(client, documents)
    else:
        print("[STEP 2/3] Skipping index creation/population.")

    # ------------------------------------------------------------------
    # Step 4 — Parse queries
    # ------------------------------------------------------------------
    print("\n[STEP 4] Parsing queries …")
    queries = parse_queries(QUERIES_XML)

    # ------------------------------------------------------------------
    # Step 5 — Parse qrels
    # ------------------------------------------------------------------
    print("\n[STEP 5] Parsing qrels …")
    qrels = parse_qrels(QRELS_TXT)

    # ------------------------------------------------------------------
    # Step 6 — Build helper layer
    # ------------------------------------------------------------------
    print("\n[STEP 6] Initialising statistics helper …")
    helper = ESHelper(client)

    # ------------------------------------------------------------------
    # Steps 7, 8, 9 — Rank with all models & write results
    # ------------------------------------------------------------------
    print("\n[STEPS 7-9] Ranking documents with all models …")
    all_results = run_all_models(queries, helper)

    # ------------------------------------------------------------------
    # Step 10 — Evaluate
    # ------------------------------------------------------------------
    print("\n[STEP 10] Running trec_eval …")
    try:
        metrics = evaluate_all(QRELS_TXT)
    except RuntimeError as e:
        print(f"[WARNING] {e}")
        metrics = {}

    # ------------------------------------------------------------------
    # Step 11 — Compare & analyse
    # ------------------------------------------------------------------
    print("\n[STEP 11] Comparing models …")
    compare_models(metrics)

    # Optional per-query comparison
    if args.query:
        compare_top10(all_results, qrels, args.query)

    print("\n✅  Pipeline complete. Results in ./results/")


if __name__ == "__main__":
    main()