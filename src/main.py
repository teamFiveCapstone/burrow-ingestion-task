import os, json, boto3
from llama_index.readers.docling import DoclingReader
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.ingestion import IngestionPipeline
from llama_index.vector_stores.opensearch import (
    OpensearchVectorStore,
    OpensearchVectorClient,
)

# Main pipeline
def main():
    # Step 1: Load envs 
    creds = json.loads(os.environ["PINECONE_API_KEY"])
    bucket_name = os.environ["S3_BUCKET_NAME"]
    s3_key = os.environ["S3_OBJECT_KEY"]
    
   # Step 2: Create presigned S3 URL
    s3 = boto3.client("s3", region_name="us-east-1")
    presigned_url = s3.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": bucket_name, "Key": s3_key},
        ExpiresIn=3600,  # 1 hour
    )

    # Step 3: Read + convert document to Markdown
    reader = DoclingReader(export_type="markdown")
    docs_md = reader.load_data(presigned_url)

    # Step 4: Parse Markdown into nodes
    node_parser = MarkdownNodeParser()
    nodes = node_parser.get_nodes_from_documents(docs_md)

    # Step 5: Set up OpenSearch client and vector store
    endpoint = "https://wh3q028h23sdc9u2hsab.us-east-1.aoss.amazonaws.com"
    index_name = "test"
    client = OpensearchVectorClient(endpoint, index_name, 1536)
    vector_store = OpensearchVectorStore(client)

    # Step 6: Embeddings model
    embed_model = OpenAIEmbedding(
        model="text-embedding-3-small",
        api_key=creds["OPENAI_API_KEY"]
    )

    # Step 7: Ingestion pipeline
    pipeline = IngestionPipeline(
        transformations=[embed_model],
        vector_store=vector_store,
    )

    # Step 8: Run pipeline
    pipeline.run(nodes=nodes)
    print("Ingestion complete — data stored in Open Search.")

# Run
if __name__ == "__main__":
    main()