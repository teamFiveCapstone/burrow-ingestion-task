import os, json
from llama_index.readers.s3 import S3Reader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

bucket_name = os.environ["S3_BUCKET_NAME"]
s3_key = os.environ["S3_OBJECT_KEY"]

# Main pipeline
def main(bucket_name, s3_key):
    # Load Pinecone + OpenAI keys from JSON env var
    creds = json.loads(os.environ["PINECONE_API_KEY"])
    
    namespace = "lion"
    index_name = "lion"

    # Step 1: Read + convert document
    reader = S3Reader(
        bucket=bucket_name,
        key=s3_key,
    )

    documents = reader.load_data()

    # Step 2: Create sentence splitter
    splitter = SentenceSplitter(
        chunk_size=800,
        chunk_overlap=120,
        include_metadata=True,
    )

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
        transformations=[
            splitter,
            embed_model,
        ],
        vector_store=vector_store,
    )

    # Step 6: Run pipeline
    nodes = pipeline.run(documents=documents)
    print(f"Pipeline produced {len(nodes)} nodes")

# Run
if __name__ == "__main__":
    main(bucket_name, s3_key)