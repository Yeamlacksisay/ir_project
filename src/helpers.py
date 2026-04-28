"""
helpers.py
----------
Step 6 — Elasticsearch statistics helper layer.

All 1400 term vectors are fetched in ONE bulk request at startup,
then everything runs from in-memory dicts — no per-query ES calls.
"""

from __future__ import annotations
from elasticsearch import Elasticsearch
from elasticsearch_setup import INDEX


class ESHelper:

    def __init__(self, client: Elasticsearch):
        self.es = client
        self._avg_dl: float | None = None
        self._total_docs: int | None = None
        self._vocab_size: int | None = None
        # {doc_id -> {term -> tf}}
        self._tv_cache: dict[str, dict[str, int]] = {}
        # {doc_id -> doc_length}
        self._dl_cache: dict[str, int] = {}
        # {term -> df}
        self._df_cache: dict[str, int] = {}
        # {term -> collection_tf}
        self._ctf_cache: dict[str, int] = {}
        self._total_tokens: int | None = None
        self._all_doc_ids: list[str] | None = None

    # ------------------------------------------------------------------
    # Bulk warm — call ONCE at startup, loads everything into RAM
    # ------------------------------------------------------------------

    def warm(self) -> None:
        """
        Fetch ALL term vectors in a single mtermvectors request.
        After this call every stat is served from memory — O(1).
        """
        if self._tv_cache:
            return  # already warm

        print("[helpers] Fetching all term vectors in bulk …")
        ids = self.get_all_doc_ids()

        # mtermvectors accepts up to 10 000 ids at once
        resp = self.es.mtermvectors(
            index=INDEX,
            ids=ids,
            fields=["body_text"],
            term_statistics=True,
            field_statistics=True,
            offsets=False,
            positions=False,
        )

        for doc in resp["docs"]:
            doc_id = doc["_id"]
            tv: dict[str, int] = {}
            terms_data = (
                doc.get("term_vectors", {})
                   .get("body_text", {})
                   .get("terms", {})
            )
            for term, stats in terms_data.items():
                tf = stats.get("term_freq", 0)
                tv[term] = tf
                # accumulate DF and collection TF while we're here
                self._df_cache[term]  = stats.get("doc_freq", 0)
                self._ctf_cache[term] = self._ctf_cache.get(term, 0) + tf

            self._tv_cache[doc_id] = tv
            self._dl_cache[doc_id] = sum(tv.values())

        # derived stats
        self._total_docs   = len(ids)
        total_len          = sum(self._dl_cache.values())
        self._avg_dl       = total_len / self._total_docs if self._total_docs else 1.0
        self._total_tokens = total_len
        all_terms          = set(self._df_cache.keys())
        self._vocab_size   = len(all_terms)

        print(f"[helpers] avg_doc_length = {self._avg_dl:.2f}  (N={self._total_docs})")
        print(f"[helpers] vocab_size     = {self._vocab_size}")
        print(f"[helpers] total_tokens   = {self._total_tokens}")

    # ------------------------------------------------------------------
    # Public API  (all O(1) after warm())
    # ------------------------------------------------------------------

    def get_all_doc_ids(self) -> list[str]:
        if self._all_doc_ids is not None:
            return self._all_doc_ids
        resp = self.es.search(
            index=INDEX,
            body={"query": {"match_all": {}}, "_source": False},
            size=10_000,
        )
        self._all_doc_ids = [hit["_id"] for hit in resp["hits"]["hits"]]
        return self._all_doc_ids

    def get_tf(self, term: str, doc_id: str) -> int:
        return self._tv_cache.get(doc_id, {}).get(term, 0)

    def get_df(self, term: str) -> int:
        return self._df_cache.get(term, 0)

    def get_doc_length(self, doc_id: str) -> int:
        return self._dl_cache.get(doc_id, 0)

    def get_avg_doc_length(self) -> float:
        if self._avg_dl is None:
            self.warm()
        return self._avg_dl  # type: ignore

    def get_total_docs(self) -> int:
        if self._total_docs is None:
            self.warm()
        return self._total_docs  # type: ignore

    def get_vocab_size(self) -> int:
        if self._vocab_size is None:
            self.warm()
        return self._vocab_size  # type: ignore

    def get_collection_tf(self, term: str) -> int:
        return self._ctf_cache.get(term, 0)

    def get_total_tokens(self) -> int:
        if self._total_tokens is None:
            self.warm()
        return self._total_tokens  # type: ignore