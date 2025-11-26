# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "boto3",
#     "llama-index-core",
#     "llama-index-embeddings-bedrock",
#     "llama-index-readers-docling",
#     "llama-index-node-parser-docling",
#     "llama-index-vector-stores-postgres",
#     "onnxruntime",
#     "psycopg2-binary",
#     "requests",
#     "transformers",
# ]
# ///

import os, boto3, psycopg2, requests
from pathlib import Path
from llama_index.readers.docling import DoclingReader
from llama_index.node_parser.docling import DoclingNodeParser
from llama_index.embeddings.bedrock import BedrockEmbedding
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core.ingestion import IngestionPipeline

from transformers import AutoTokenizer
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer

# ---------- CONFIG ----------

BUCKET_NAME = os.environ["S3_BUCKET_NAME"]
S3_KEY = os.environ["S3_OBJECT_KEY"]
TABLE_NAME = "burrow_table_hybrid2"
EMBED_DIM = 1024  
INGESTION_API_TOKEN = os.environ["INGESTION_API_TOKEN"]
DOCUMENT_ID = Path(S3_KEY).stem
print(DOCUMENT_ID)

DB_HOST = os.environ["DB_HOST"]
DB_PORT = os.environ["DB_PORT"]
DB_NAME = os.environ["DB_NAME"]
DB_USER = os.environ["DB_USER"]
DB_PASSWORD = os.environ["DB_PASSWORD"]

ALB_BASE_URL = os.environ["ALB_BASE_URL"]

MAX_TOKENS = 4096
TOKENIZER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ---------- HELPERS ----------

def ensure_pgvector_extension():
    print("Ensuring pgvector extension is installed")
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        connect_timeout=60,
    )
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cur.close()
    conn.close()
    print("pgvector extension ready.\n")

def update_document_status(status):
    url = f"{ALB_BASE_URL}/api/documents/{DOCUMENT_ID}"
    headers = {"x-api-token": INGESTION_API_TOKEN }
    data = {"status": status}

    print(f"[PATCH] {url} → {status}")
    resp = requests.patch(url, headers=headers, json=data, timeout=60)
    print(f"[PATCH] Status Code: {resp.status_code}\nResponse Body: {resp.text}")
    resp.raise_for_status()

def create_hybrid_chunker():
    print(f"Creating HybridChunker with max_tokens={MAX_TOKENS}")

    tokenizer = HuggingFaceTokenizer(
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL),
        max_tokens = MAX_TOKENS
    )

    chunker = HybridChunker(
        tokenizer = tokenizer,
        merge_peers = True
    )

    print("HybridChunker initialized")
    return chunker

# ---------- MAIN INGESTION LOGIC ----------

def main():
    # Step 1: Call helpers
    ensure_pgvector_extension()

    # Step 2: Create presigned S3 URL
    s3 = boto3.client("s3", region_name="us-east-1")
    presigned_url = s3.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": BUCKET_NAME, "Key": S3_KEY},
        ExpiresIn=3600,
    )

    # Step 3: Read document
    reader = DoclingReader()
    docs = reader.load_data(presigned_url)

    # Step 4: Parse document into nodes
    hybrid_chunker = create_hybrid_chunker()

    node_parser = DoclingNodeParser(chunker=hybrid_chunker)
    nodes = node_parser.get_nodes_from_documents(docs)

    # Step 5: Initialize Aurora-backed PGVectorStore
    vector_store = PGVectorStore.from_params(
        database=DB_NAME,
        host=DB_HOST,
        password=DB_PASSWORD,
        port=DB_PORT,
        user=DB_USER,
        table_name=TABLE_NAME,
        embed_dim=EMBED_DIM,  
        hybrid_search=True,
        text_search_config="english",
        hnsw_kwargs={
            "hnsw_m": 16,
            "hnsw_ef_construction": 64,
            "hnsw_ef_search": 40,
            "hnsw_dist_method": "vector_cosine_ops",
        },
    )

    # Step 6: Bedrock embedding model (Amazon Titan v2)
    embed_model = BedrockEmbedding(
        model_name="amazon.titan-embed-text-v2:0",
        region_name="us-east-1",
    )

    # Step 7: Ingestion pipeline
    pipeline = IngestionPipeline(
        transformations=[embed_model],
        vector_store=vector_store,
    )

    # Step 8: Run pipeline
    pipeline.run(nodes=nodes, num_workers=2)
    print("Ingestion complete — data stored in Aurora (pgvector).")

# ---------- Wrapper ----------

def main_with_status():
    try:
        update_document_status("running")
        print('Document running')
    except Exception as e:
        print(f"[status] WARNING: failed to set status=running: {e}")

    try:
        main()
    except Exception as e:
        print(f"[ingestion] ERROR: {e}")
        try:
            update_document_status("failed")
            print('Document failed')
        except Exception as e2:
            print(f"[status] WARNING: failed to set status=failed: {e2}")
        raise
    else:
        try:
            update_document_status("finished")
            print('Document finished')
        except Exception as e:
            print(f"[status] WARNING: failed to set status=finished: {e}")

# ---------- Work Flow ----------

if __name__ == "__main__":
    main_with_status()
