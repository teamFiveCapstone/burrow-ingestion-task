import os, json, boto3, psycopg2
from llama_index.readers.docling import DoclingReader
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core.ingestion import IngestionPipeline

# Aurora DB CONFIG (dev)
DB_HOST = "burrow-zach.cluster-cwxgyacqyoae.us-east-1.rds.amazonaws.com"
DB_PORT = 5432
DB_NAME = "postgres"
DB_USER = "postgres"
DB_PASSWORD = "password"  # dev only

# Helpers: Create extensions for postgres
def ensure_pgvector_extension():
    """Idempotently install pgvector extension (vector) if missing."""
    print("Ensuring pgvector extension is installed...")
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        connect_timeout=60,
    )
    conn.autocommit = True  # required for CREATE EXTENSION
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cur.close()
    conn.close()
    print("pgvector extension ready.\n")

# MAIN PIPELINE
def main():
    # Step 0: Call helpers
    ensure_pgvector_extension()

    # Step 1: Load environment variables
    bucket_name = os.environ["S3_BUCKET_NAME"]
    s3_key = os.environ["S3_OBJECT_KEY"]
    creds = json.loads(os.environ["PINECONE_API_KEY"])
    table_name = "burrow_table"      # llamaindex makes this data_burrow_table
    embed_dim = 1536   

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
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        table_name=table_name,
        embed_dim=embed_dim,
    )

    # Step 6: OpenAI embedding model for ingestion
    embed_model = OpenAIEmbedding(
        model="text-embedding-3-small",
        api_key=creds["OPENAI_API_KEY"],
    )

    # Step 7: Ingestion pipeline
    pipeline = IngestionPipeline(
        transformations=[embed_model],
        vector_store=vector_store,
    )

    # Step 8: Run pipeline
    pipeline.run(nodes=nodes)
    print("Ingestion complete — data stored in Aurora (pgvector).")

# ENTRYPOINT
if __name__ == "__main__":
    main()
