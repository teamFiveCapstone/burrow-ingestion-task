# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "boto3",
#     "llama-index-core",
#     "llama-index-embeddings-bedrock",
#     "llama-index-readers-docling",
#     "llama-index-vector-stores-postgres",
#     "onnxruntime",
#     "psycopg2-binary",
#     "requests",
# ]
# ///

import os, json, boto3, psycopg2, requests
from llama_index.readers.docling import DoclingReader
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.embeddings.bedrock import BedrockEmbedding
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core.ingestion import IngestionPipeline

# ---------- CONFIG ----------

bucket_name = os.environ["S3_BUCKET_NAME"]
s3_key = os.environ["S3_OBJECT_KEY"]
table_name = "burrow_table_hybrid2"
embed_dim = 1024  
INGESTION_API_TOKEN = os.environ["INGESTION_API_TOKEN"]
document_id = s3_key.split('.')[0]
print(document_id)

DB_HOST = "burrow-serverless-wilson.cluster-cwxgyacqyoae.us-east-1.rds.amazonaws.com"
DB_PORT = 5432
DB_NAME = "embeddings"
DB_USER = "burrow"
DB_PASSWORD = "capstone"  # dev only

AWS_ALB_URL = "http://rag-lb-970809826.us-east-1.elb.amazonaws.com"

# ---------- HELPERS ----------

def ensure_pgvector_extension_and_drop_old():
    print("Ensuring pgvector extension is installed, dropping old tables...")
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

def update_document_status(document_id, status):
    url = f"{AWS_ALB_URL}/api/documents/{document_id}"
    headers = {"x-api-token": INGESTION_API_TOKEN }
    data = {"status": status}

    print(f"[PATCH] {url} → {status}")
    resp = requests.patch(url, headers=headers, json=data, timeout=60)
    print(f"[PATCH] Status Code: {resp.status_code}\nResponse Body: {resp.text}")
    resp.raise_for_status()

# ---------- MAIN INGESTION LOGIC ----------

def main():
    # Step 1: Call helpers
    ensure_pgvector_extension_and_drop_old()

    # Step 2: Create presigned S3 URL
    s3 = boto3.client("s3", region_name="us-east-1")
    presigned_url = s3.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": bucket_name, "Key": s3_key},
        ExpiresIn=3600,
    )

    # Step 3: Read + convert document to Markdown
    reader = DoclingReader(export_type="markdown")
    docs_md = reader.load_data(presigned_url)

    # Step 4: Parse Markdown into nodes
    node_parser = MarkdownNodeParser()
    nodes = node_parser.get_nodes_from_documents(docs_md)

    # Step 5: Initialize Aurora-backed PGVectorStore
    vector_store = PGVectorStore.from_params(
        database=DB_NAME,
        host=DB_HOST,
        password=DB_PASSWORD,
        port=DB_PORT,
        user=DB_USER,
        table_name=table_name,
        embed_dim=embed_dim,  
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
        update_document_status(document_id, "running")
        print('Document running')
    except Exception as e:
        print(f"[status] WARNING: failed to set status=running: {e}")

    try:
        main()
    except Exception as e:
        print(f"[ingestion] ERROR: {e}")
        try:
            update_document_status(document_id, "failed")
            print('Document failed')
        except Exception as e2:
            print(f"[status] WARNING: failed to set status=failed: {e2}")
        raise
    else:
        try:
            update_document_status(document_id, "finished")
            print('Document finished')
        except Exception as e:
            print(f"[status] WARNING: failed to set status=finished: {e}")

# ---------- Work Flow ----------

if __name__ == "__main__":
    main_with_status()
