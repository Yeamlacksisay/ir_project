"""
elasticsearch_setup.py
----------------------
Steps 2 & 3 — Create the Elasticsearch index and bulk-index all documents.
"""

from elasticsearch import Elasticsearch, helpers

ES_HOST     = "https://localhost:9200"
ES_USER     = "elastic"
ES_PASSWORD = "ee-uGx2NrFldkwoHf3Lp"   # from ES startup output
INDEX       = "cranfield"


def get_client() -> Elasticsearch:
    """Return a connected Elasticsearch client (HTTPS + basic auth)."""
    client = Elasticsearch(
        ES_HOST,
        basic_auth=(ES_USER, ES_PASSWORD),
        verify_certs=False,   # self-signed cert — safe for local dev
        ssl_show_warn=False,
    )
    if not client.ping():
        raise ConnectionError(
            f"Cannot reach Elasticsearch at {ES_HOST}.\n"
            "Make sure elasticsearch.bat is running."
        )
    return client


INDEX_SETTINGS = {
    "settings": {
        "number_of_shards":   1,
        "number_of_replicas": 0,
    },
    "mappings": {
        "properties": {
            "docno": {
                "type": "keyword"
            },
            "body_text": {
                "type":        "text",
                "analyzer":    "standard",
                "term_vector": "with_positions_offsets",
            },
        }
    },
}


def create_index(client: Elasticsearch, force_recreate: bool = False) -> None:
    if client.indices.exists(index=INDEX):
        if force_recreate:
            client.indices.delete(index=INDEX)
            print(f"[setup] Deleted existing index '{INDEX}'")
        else:
            print(f"[setup] Index '{INDEX}' already exists — skipping creation.")
            return

    client.indices.create(index=INDEX, body=INDEX_SETTINGS)
    print(f"[setup] Index '{INDEX}' created successfully.")


def index_documents(client: Elasticsearch, documents: list[dict]) -> None:
    def _actions():
        for doc in documents:
            yield {
                "_index": INDEX,
                "_id":    doc["docno"],
                "_source": {
                    "docno":     doc["docno"],
                    "body_text": doc["body_text"],
                },
            }

    successes, errors = helpers.bulk(client, _actions(), raise_on_error=False)
    print(f"[setup] Indexed {successes} documents. Errors: {len(errors)}")
    client.indices.refresh(index=INDEX)