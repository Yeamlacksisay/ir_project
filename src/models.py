"""
models.py
---------
Step 7 — Manual implementations of five IR ranking models.

Each scoring function takes:
    query_terms : list[str]   — already tokenised & cleaned query
    doc_id      : str         — ES document ID (== docno)
    helper      : ESHelper    — statistics engine from helpers.py

And returns a single float score (higher = more relevant).

Models implemented
------------------
1. TF-IDF         — classic weighted term frequency
2. Okapi TF       — normalised TF (no IDF component)
3. BM25           — Robertson/Spärck Jones probabilistic model  ⭐
4. Laplace LM     — language model with additive (+1) smoothing
5. Jelinek-Mercer — language model with linear interpolation smoothing
"""

import math
from helpers import ESHelper


# ===========================================================================
# 1. TF-IDF
# ===========================================================================

def score_tfidf(query_terms: list[str], doc_id: str, helper: ESHelper) -> float:
    """
    Classic TF-IDF.

        score = Σ  tf(t,d) × log( N / df(t) )

    where:
        tf(t,d) = raw term frequency in document d
        N       = total number of documents
        df(t)   = document frequency of term t
    """
    N = helper.get_total_docs()
    score = 0.0

    for term in set(query_terms):   # set to avoid double-counting repeated query terms
        tf = helper.get_tf(term, doc_id)
        if tf == 0:
            continue
        df = helper.get_df(term)
        if df == 0:
            continue
        idf = math.log(N / df)
        score += tf * idf

    return score


# ===========================================================================
# 2. Okapi TF  (normalised TF, no IDF)
# ===========================================================================

def score_okapi_tf(query_terms: list[str], doc_id: str, helper: ESHelper,
                   k1: float = 1.2, b: float = 0.75) -> float:
    """
    Okapi TF — length-normalised term frequency without IDF.

        score = Σ  tf_norm(t,d)

    where:
        tf_norm = tf / ( tf + k1 × ( 1 - b + b × dl/avgdl ) )

    Parameters
    ----------
    k1  : saturation parameter  (typical: 1.2)
    b   : length normalisation  (typical: 0.75)
    """
    dl    = helper.get_doc_length(doc_id)
    avgdl = helper.get_avg_doc_length()
    score = 0.0

    for term in set(query_terms):
        tf = helper.get_tf(term, doc_id)
        if tf == 0:
            continue
        normaliser = k1 * (1 - b + b * dl / avgdl)
        tf_norm = tf / (tf + normaliser)
        score += tf_norm

    return score


# ===========================================================================
# 3. BM25  ⭐
# ===========================================================================

def score_bm25(query_terms: list[str], doc_id: str, helper: ESHelper,
               k1: float = 1.5, b: float = 0.75, k3: float = 500) -> float:
    """
    Okapi BM25.

        score = Σ  IDF(t) × [ tf_norm(t,d) × (k1+1) ] / [ tf_norm(t,d) + k1×(1-b+b×dl/avgdl) ]
                          ×  [ (k3+1)×qtf ] / [ k3+qtf ]

    where:
        IDF(t)  = log( (N - df + 0.5) / (df + 0.5) )   Robertson IDF
        qtf     = query term frequency (1 for single-term queries; kept for completeness)
        k3      = query-term saturation (typically 500-1000)

    Parameters
    ----------
    k1 : document TF saturation   (typical: 1.2-2.0)
    b  : length normalisation     (typical: 0.75)
    k3 : query TF saturation      (typical: 500)
    """
    N     = helper.get_total_docs()
    dl    = helper.get_doc_length(doc_id)
    avgdl = helper.get_avg_doc_length()
    score = 0.0

    # Count query term frequencies (qtf)
    qtf_map: dict[str, int] = {}
    for term in query_terms:
        qtf_map[term] = qtf_map.get(term, 0) + 1

    for term, qtf in qtf_map.items():
        tf = helper.get_tf(term, doc_id)
        df = helper.get_df(term)
        if df == 0:
            continue

        # Robertson IDF — clipped at 0 to avoid negative scores
        idf = max(0.0, math.log((N - df + 0.5) / (df + 0.5)))

        # Normalised document TF
        tf_norm = tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))

        # Query TF component
        qtf_weight = (k3 + 1) * qtf / (k3 + qtf)

        score += idf * tf_norm * qtf_weight

    return score


# ===========================================================================
# 4. Laplace Language Model  (additive / +1 smoothing)
# ===========================================================================

def score_laplace(query_terms: list[str], doc_id: str, helper: ESHelper) -> float:
    """
    Query-likelihood language model with Laplace (add-1) smoothing.

        P(t|d) = ( tf(t,d) + 1 ) / ( dl + V )
        score  = Σ  log P(t|d)

    where:
        V  = vocabulary size (distinct terms in the collection)
        dl = document length
    """
    dl = helper.get_doc_length(doc_id)
    V  = helper.get_vocab_size()
    score = 0.0

    for term in query_terms:
        tf   = helper.get_tf(term, doc_id)
        prob = (tf + 1) / (dl + V)
        score += math.log(prob)

    return score


# ===========================================================================
# 5. Jelinek-Mercer Language Model  (linear interpolation smoothing)
# ===========================================================================

def score_jelinek_mercer(query_terms: list[str], doc_id: str,
                         helper: ESHelper, lam: float = 0.4) -> float:
    """
    Query-likelihood language model with Jelinek-Mercer smoothing.

        P(t|d) = (1-λ) × tf(t,d)/dl  +  λ × ctf(t)/C
        score  = Σ  log P(t|d)

    where:
        λ    = smoothing parameter  (0 < λ < 1; typical: 0.1–0.5)
        cft  = collection term frequency of t
        C    = total tokens in collection

    Parameters
    ----------
    lam : smoothing weight towards the collection model (default 0.4)
    """
    dl = helper.get_doc_length(doc_id)
    C  = helper.get_total_tokens()

    if dl == 0 or C == 0:
        return float("-inf")

    score = 0.0
    for term in query_terms:
        tf  = helper.get_tf(term, doc_id)
        cft = helper.get_collection_tf(term)

        p_doc        = tf  / dl  if dl > 0 else 0.0
        p_collection = cft / C   if C  > 0 else 1e-10   # floor to avoid /0

        p_smooth = (1 - lam) * p_doc + lam * p_collection

        if p_smooth <= 0:
            score += math.log(1e-10)   # large negative penalty
        else:
            score += math.log(p_smooth)

    return score