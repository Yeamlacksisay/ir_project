import re
from lxml import etree

def _load_xml_root(filepath: str):
    with open(filepath, "rb") as fh:
        raw = fh.read()
    return etree.fromstring(raw)

# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Lowercase, strip extra whitespace and non-alphanumeric chars."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)   # keep letters, digits, spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Step 1 — Parse Documents
# ---------------------------------------------------------------------------

def parse_documents(filepath: str) -> list[dict]:
    """
    Parse the Cranfield XML document collection.

    Returns
    -------
    List of dicts with keys: docno (str), title (str), text (str)
    """
    root = _load_xml_root(filepath)


    documents = []
    for doc in root.findall(".//doc"):
        docno_el = doc.find("docno")
        title_el = doc.find("title")
        text_el  = doc.find("text")

        docno = docno_el.text.strip() if docno_el is not None and docno_el.text else ""
        title = title_el.text.strip() if title_el is not None and title_el.text else ""
        text  = text_el.text.strip()  if text_el  is not None and text_el.text  else ""

        # Combine title + body so ranking has the full content
        combined = f"{title} {text}"

        documents.append({
            "docno":     docno,
            "title":     clean_text(title),
            "body_text": clean_text(combined),
        })

    print(f"[parser] Loaded {len(documents)} documents from '{filepath}'")
    return documents


# ---------------------------------------------------------------------------
# Step 4 — Parse Queries
# ---------------------------------------------------------------------------

def parse_queries(filepath: str) -> list[dict]:
    """
    Parse the XML query file.

    Returns
    -------
    List of dicts with keys: qid (str), text (str)
    """
    root = _load_xml_root(filepath)

    queries = []
    for top in root.findall(".//top"):
        num_el   = top.find("num")
        title_el = top.find("title")

        qid  = num_el.text.strip()   if num_el   is not None and num_el.text   else ""
        text = title_el.text.strip() if title_el is not None and title_el.text else ""

        queries.append({
            "qid":  qid,
            "text": clean_text(text),
        })

    print(f"[parser] Loaded {len(queries)} queries from '{filepath}'")
    return queries


# ---------------------------------------------------------------------------
# Step 5 — Parse QRELS
# ---------------------------------------------------------------------------

def parse_qrels(filepath: str) -> dict[str, set[str]]:
    """
    Parse a TREC-format qrels file.

    Each line:  query_id  0  doc_id  relevance
    A document is considered relevant if relevance >= 1.

    Returns
    -------
    Dict mapping query_id -> set of relevant doc_ids
    """
    qrels: dict[str, set[str]] = {}

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            qid, _, docno, relevance = parts[0], parts[1], parts[2], parts[3]
            if int(relevance) >= 1:
                qrels.setdefault(qid, set()).add(docno)

    print(f"[parser] Loaded qrels for {len(qrels)} queries from '{filepath}'")
    return qrels