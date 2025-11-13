import os, json, boto3
from llama_index.readers.docling import DoclingReader
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.ingestion import IngestionPipeline
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# Create variables from event bridge
bucket_name = os.environ["S3_BUCKET_NAME"]
s3_key = os.environ["S3_OBJECT_KEY"]

# Create presigned S3 URL
s3 = boto3.client("s3", region_name="us-east-1")
presigned_url = s3.generate_presigned_url(
    ClientMethod="get_object",
    Params={"Bucket": bucket_name, "Key": s3_key},
    ExpiresIn=3600,  # 1 hour
)

# Main pipeline
def main(bucket_name, s3_key):
    # Load Pinecone + OpenAI keys from JSON env var
    creds = json.loads(os.environ["PINECONE_API_KEY"])
    
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
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index, namespace=namespace)

    # Step 4: Embeddings model
    embed_model = OpenAIEmbedding(
        model="text-embedding-3-small",
        api_key=creds["OPENAI_API_KEY"]
    )

    # Step 5: Ingestion pipeline
    pipeline = IngestionPipeline(
        transformations=[embed_model],
        vector_store=vector_store,
    )

    # Step 6: Run pipeline
    pipeline.run(nodes=nodes)
    print("Ingestion complete — data stored in Pinecone.")

# Run
if __name__ == "__main__":
    main(bucket_name, s3_key)