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
import os
import boto3
import psycopg2
import requests
from pathlib import Path
from llama_index.readers.docling import DoclingReader
from llama_index.node_parser.docling import DoclingNodeParser
from llama_index.embeddings.bedrock import BedrockEmbedding
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core.ingestion import IngestionPipeline
from transformers import AutoTokenizer
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from logger import log_info, log_exception

BUCKET_NAME = os.environ["S3_BUCKET_NAME"]
S3_KEY = os.environ["S3_OBJECT_KEY"]
TABLE_NAME = "burrow_table_hybrid2"
EMBED_DIM = 1024
INGESTION_API_TOKEN = os.environ["INGESTION_API_TOKEN"]
DOCUMENT_ID = Path(S3_KEY).stem
DB_HOST = os.environ["DB_HOST"]
DB_PORT = os.environ["DB_PORT"]
DB_NAME = os.environ["DB_NAME"]
DB_USER = os.environ["DB_USER"]
DB_PASSWORD = os.environ["DB_PASSWORD"]
ALB_BASE_URL = os.environ["ALB_BASE_URL"]
EVENT_TYPE = os.environ.get("EVENT_TYPE", "Object Created")
MAX_TOKENS = 4096
TOKENIZER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
ORIGIN_VERIFY_TOKEN = os.environ["ORIGIN_VERIFY_TOKEN"]
log_info(ORIGIN_VERIFY_TOKEN)

log_info(
    "Ingestion script loaded",
    document_id=DOCUMENT_ID,
    bucket=BUCKET_NAME,
    key=S3_KEY,
    table_name=TABLE_NAME,
    event_type=EVENT_TYPE,
)


def ensure_pgvector_extension():
    log_info(
        "Ensuring pgvector extension is installed",
        document_id=DOCUMENT_ID,
        db_host=DB_HOST,
        db_name=DB_NAME,
    )
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
    log_info("pgvector extension ready", document_id=DOCUMENT_ID)


def update_document_status(status):
    url = f"{ALB_BASE_URL}/api/documents/{DOCUMENT_ID}"
    headers = {
        "x-api-token": INGESTION_API_TOKEN,
        "X-Origin-Verify": ORIGIN_VERIFY_TOKEN,
    }
    data = {"status": status}

    log_info(
        "Updating document status via management-api",
        document_id=DOCUMENT_ID,
        status=status,
        url=url,
    )

    resp = requests.patch(url, headers=headers, json=data, timeout=60)

    log_info(
        "Document status update response",
        document_id=DOCUMENT_ID,
        status=status,
        http_status=resp.status_code,
        response_body=resp.text[:500],
    )

    resp.raise_for_status()


def create_hybrid_chunker():
    log_info(
        "Creating HybridChunker",
        document_id=DOCUMENT_ID,
        max_tokens=MAX_TOKENS,
        tokenizer_model=TOKENIZER_MODEL,
    )

    tokenizer = HuggingFaceTokenizer(
        tokenizer=AutoTokenizer.from_pretrained(TOKENIZER_MODEL),
        max_tokens=MAX_TOKENS,
    )

    chunker = HybridChunker(
        tokenizer=tokenizer,
        merge_peers=True,
    )

    log_info("HybridChunker initialized", document_id=DOCUMENT_ID)
    return chunker


def delete_embeddings_for_document():
    log_info(
        "Deleting embeddings for document",
        document_id=DOCUMENT_ID,
        table_name=TABLE_NAME,
    )

    embed_model = BedrockEmbedding(
        model_name="amazon.titan-embed-text-v2:0",
        region_name="us-east-1",
    )

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
    )

    from llama_index.core import VectorStoreIndex

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )

    try:
        log_info(
            "Calling delete_ref_doc",
            document_id=DOCUMENT_ID,
        )
        index.delete_ref_doc(DOCUMENT_ID, delete_from_docstore=True)
        log_info(
            "delete_ref_doc completed",
            document_id=DOCUMENT_ID,
        )
    except Exception:
        log_exception(
            "delete_ref_doc failed, falling back to SQL delete",
            document_id=DOCUMENT_ID,
            table_name=TABLE_NAME,
        )
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
        )
        cur = conn.cursor()
        sql = f"DELETE FROM data_{TABLE_NAME} WHERE metadata->>'file_name' LIKE %s"
        cur.execute(sql, (f"%{DOCUMENT_ID}%",))
        deleted_count = cur.rowcount
        conn.commit()
        cur.close()
        conn.close()
        log_info(
            "SQL fallback delete completed",
            document_id=DOCUMENT_ID,
            deleted_count=deleted_count,
        )


def main():
    log_info(
        "Starting ingestion",
        document_id=DOCUMENT_ID,
        bucket=BUCKET_NAME,
        key=S3_KEY,
        table_name=TABLE_NAME,
    )

    ensure_pgvector_extension()

    s3 = boto3.client("s3", region_name="us-east-1")
    presigned_url = s3.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": BUCKET_NAME, "Key": S3_KEY},
        ExpiresIn=3600,
    )
    log_info(
        "Generated presigned S3 URL",
        document_id=DOCUMENT_ID,
        bucket=BUCKET_NAME,
    )

    reader = DoclingReader(export_type=DoclingReader.ExportType.JSON)
    docs = reader.load_data(presigned_url)
    log_info(
        "Loaded document from S3",
        document_id=DOCUMENT_ID,
        doc_count=len(docs),
    )

    for doc in docs:
        doc.id_ = DOCUMENT_ID

    hybrid_chunker = create_hybrid_chunker()
    node_parser = DoclingNodeParser(chunker=hybrid_chunker)
    nodes = node_parser.get_nodes_from_documents(docs)
    log_info(
        "Parsed document into nodes",
        document_id=DOCUMENT_ID,
        node_count=len(nodes),
    )

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
    log_info(
        "Initialized PGVectorStore for ingestion",
        document_id=DOCUMENT_ID,
        table_name=TABLE_NAME,
    )

    embed_model = BedrockEmbedding(
        model_name="amazon.titan-embed-text-v2:0",
        region_name="us-east-1",
    )

    pipeline = IngestionPipeline(
        transformations=[embed_model],
        vector_store=vector_store,
    )

    pipeline.run(nodes=nodes, num_workers=2)
    log_info(
        "Ingestion complete — data stored in Aurora (pgvector)",
        document_id=DOCUMENT_ID,
        node_count=len(nodes),
    )


def main_with_status():
    is_delete = EVENT_TYPE == "Object Deleted"

    if is_delete:
        log_info(
            "Delete event received",
            document_id=DOCUMENT_ID,
            event_type=EVENT_TYPE,
            bucket=BUCKET_NAME,
            key=S3_KEY,
        )

        try:
            update_document_status("deleting")
        except Exception:
            log_exception(
                "Failed to set status=deleting",
                document_id=DOCUMENT_ID,
            )

        try:
            delete_embeddings_for_document()
            update_document_status("deleted")
            log_info(
                "Delete flow completed successfully",
                document_id=DOCUMENT_ID,
            )
        except Exception:
            log_exception(
                "Delete flow failed",
                document_id=DOCUMENT_ID,
            )
            try:
                update_document_status("delete_failed")
            except Exception:
                log_exception(
                    "Failed to set status=delete_failed",
                    document_id=DOCUMENT_ID,
                )
            raise

    else:
        log_info(
            "Create/ingestion event received",
            document_id=DOCUMENT_ID,
            event_type=EVENT_TYPE,
            bucket=BUCKET_NAME,
            key=S3_KEY,
        )

        try:
            update_document_status("running")
            log_info(
                "Document marked as running",
                document_id=DOCUMENT_ID,
            )
        except Exception:
            log_exception(
                "Failed to set status=running",
                document_id=DOCUMENT_ID,
            )

        try:
            main()
        except Exception:
            log_exception(
                "Ingestion failed",
                document_id=DOCUMENT_ID,
            )
            try:
                update_document_status("failed")
                log_info(
                    "Document marked as failed",
                    document_id=DOCUMENT_ID,
                )
            except Exception:
                log_exception(
                    "Failed to set status=failed",
                    document_id=DOCUMENT_ID,
                )
            raise
        else:
            try:
                update_document_status("finished")
                log_info(
                    "Document marked as finished",
                    document_id=DOCUMENT_ID,
                )
            except Exception:
                log_exception(
                    "Failed to set status=finished",
                    document_id=DOCUMENT_ID,
                )


if __name__ == "__main__":
    log_info(
        "Ingestion task starting",
        document_id=DOCUMENT_ID,
        event_type=EVENT_TYPE,
        bucket=BUCKET_NAME,
        key=S3_KEY,
        table_name=TABLE_NAME,
    )
    main_with_status()
