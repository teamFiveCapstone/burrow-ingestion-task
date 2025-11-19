import os
import json
import boto3
from dotenv import load_dotenv

import psycopg2
from openai import OpenAI

from llama_index.readers.docling import DoclingReader
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.ingestion import IngestionPipeline
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec


# ------------------------------------------------------------
# Aurora DB CONFIG (hard-coded for dev)
# ------------------------------------------------------------
DB_HOST = "burrow-zach.cluster-cwxgyacqyoae.us-east-1.rds.amazonaws.com"
DB_PORT = 5432
DB_NAME = "postgres"
DB_USER = "postgres"
DB_PASSWORD = "password"     # dev only, don't commit to public repos


# ------------------------------------------------------------
# SMOKE TEST: Aurora DB connection
# ------------------------------------------------------------
def test_db_connection():
    print(f"\nTesting DB connection to {DB_HOST}:{DB_PORT}/{DB_NAME} ...")

    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        connect_timeout=60
    )

    cur = conn.cursor()
    cur.execute("SELECT 1;")
    row = cur.fetchone()

    print("DB test row:", row)
    cur.close()
    conn.close()

    print("DB connection OK!\n")


# ------------------------------------------------------------
# SMOKE TEST: OpenAI connectivity
# ------------------------------------------------------------
def test_openai_connection(creds: dict):
    print("Testing OpenAI embeddings...")

    client = OpenAI(api_key=creds["OPENAI_API_KEY"])
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input="hello from ECS ingestion task",
    )

    emb = resp.data[0].embedding
    print("OpenAI embedding length:", len(emb))
    print("OpenAI call OK!\n")


# ------------------------------------------------------------
# Load environment variables
# ------------------------------------------------------------
load_dotenv()

bucket_name = os.environ["S3_BUCKET_NAME"]
s3_key = os.environ["S3_OBJECT_KEY"]

print(f"bucket_name: {bucket_name}")
print(f"s3_key: {s3_key}")

# Create presigned S3 URL
s3 = boto3.client("s3", region_name="us-east-1")
presigned_url = s3.generate_presigned_url(
    ClientMethod="get_object",
    Params={"Bucket": bucket_name, "Key": s3_key},
    ExpiresIn=3600,
)


# ------------------------------------------------------------
# MAIN PIPELINE
# ------------------------------------------------------------
def main(bucket_name, s3_key):

    # Load Pinecone + OpenAI keys from your JSON env var
    creds = json.loads(os.environ["PINECONE_API_KEY"])

    # Run smoke tests BEFORE ingestion pipeline
    test_db_connection()
    test_openai_connection(creds)

    namespace = "lion"
    index_name = "lion"

    # Step 1: Read + convert document to Markdown
    reader = DoclingReader(export_type="markdown")
    docs_md = reader.load_data(presigned_url)

    # Step 2: Parse Markdown into nodes
    node_parser = MarkdownNodeParser()
    nodes = node_parser.get_nodes_from_documents(docs_md)

    # Step 3: Initialize Pinecone
    pc = Pinecone(api_key=creds["PINECONE_API_KEY"])
    try:
        pc.create_index(
            index_name,
            dimension=1536,
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    except Exception:
        print("Index already exists — skipping creation.")

    pinecone_index = pc.Index(index_name)
    vector_store = PineconeVectorStore(
        pinecone_index=pinecone_index,
        namespace=namespace
    )

    # Step 4: OpenAI embedding model for ingestion
    embed_model = OpenAIEmbedding(
        model="text-embedding-3-small",
        api_key=creds["OPENAI_API_KEY"],
    )

    # Step 5: Ingestion pipeline
    pipeline = IngestionPipeline(
        transformations=[embed_model],
        vector_store=vector_store,
    )

    # Step 6: Run pipeline
    pipeline.run(nodes=nodes)
    print("Ingestion complete — data stored in Pinecone.")


# ------------------------------------------------------------
# ENTRYPOINT
# ------------------------------------------------------------
if __name__ == "__main__":
    main(bucket_name, s3_key)
