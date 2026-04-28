# Cranfield IR Project

A complete Information Retrieval pipeline on the Cranfield collection using Elasticsearch and five hand-implemented ranking models.

---

## Models Implemented

| # | Model | Key Idea |
|---|-------|----------|
| 1 | **TF-IDF** | Term frequency × inverse document frequency |
| 2 | **Okapi TF** | Length-normalised TF (no IDF) |
| 3 | **BM25** ⭐ | Probabilistic model with saturation & normalisation |
| 4 | **Laplace LM** | Language model with add-1 smoothing |
| 5 | **Jelinek-Mercer LM** | Language model with linear interpolation |

---

## Project Structure

```
ir_project/
├── data/
│   ├── cranfield_docs.xml   ← document collection
│   ├── queries.xml          ← query topics
│   └── qrels.txt            ← relevance judgements (TREC format)
├── src/
│   ├── parser.py            ← Steps 1, 4, 5
│   ├── elasticsearch_setup.py ← Steps 2, 3
│   ├── helpers.py           ← Step 6  (stats engine)
│   ├── models.py            ← Step 7  (5 ranking models)
│   ├── ranker.py            ← Steps 8, 9
│   └── evaluator.py         ← Steps 10, 11
├── results/                 ← TREC-format output files
├── main.py                  ← Master pipeline
├── docker-compose.yml       ← Elasticsearch setup
└── requirements.txt
```

---

## Quick Start

### 1. Start Elasticsearch

```bash
docker compose up -d
```

Elasticsearch will be available at `http://localhost:9200`.

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your data files

Place your files in `data/`:

```
data/cranfield_docs.xml
data/queries.xml
data/qrels.txt
```

### 4. Run the full pipeline

```bash
python main.py
```

### Useful flags

```bash
python main.py --reindex          # delete & recreate ES index
python main.py --skip-index       # skip indexing (index already populated)
python main.py --eval-only        # jump to evaluation only
python main.py --query 1          # show top-10 comparison for query ID 1
```

---

## Evaluation

Install `trec_eval`:

```bash
git clone https://github.com/usnistgov/trec_eval.git
cd trec_eval && make
sudo cp trec_eval /usr/local/bin/
```

`trec_eval` runs automatically as part of `main.py`. Results are printed to the terminal.

---

## Expected XML Formats

**Documents (`cranfield_docs.xml`)**
```xml
<docs>
  <doc>
    <docno>1</docno>
    <title>Some Title</title>
    <text>Body text here...</text>
  </doc>
</docs>
```

**Queries (`queries.xml`)**
```xml
<topics>
  <top>
    <num>1</num>
    <title>query text here</title>
  </top>
</topics>
```

**QRELS (`qrels.txt`)** — standard TREC format:
```
1 0 184 2
1 0 29  1
...
```
`query_id  0  doc_id  relevance` — relevance ≥ 1 is considered relevant.

---

## Results

After running, ranked result files appear in `results/`:

```
results/tfidf.txt
results/okapi_tf.txt
results/bm25.txt
results/laplace.txt
results/jm.txt
```

Each file is in TREC format:
```
query-id Q0 docno rank score model_name
```
