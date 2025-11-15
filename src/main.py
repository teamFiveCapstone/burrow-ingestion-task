import os, json, boto3
from llama_index.readers.docling import DoclingReader
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.ingestion import IngestionPipeline
from llama_index.vector_stores.opensearch import (
    OpensearchVectorStore,
    OpensearchVectorClient,
)
from opensearchpy import RequestsHttpConnection, AWSV4SignerAuth, OpenSearch


def ensure_index(endpoint, awsauth, index_name, dim, region):
    os_client = OpenSearch(
        hosts=[{"host": endpoint.replace("https://", ""), "port": 443}],
        http_auth=awsauth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        timeout=60,
    )

    if os_client.indices.exists(index=index_name):
        print(f"Index '{index_name}' already exists, skipping creation.")
        return

    body = {
        "settings": {
            "index.knn": True
        },
        "mappings": {
            "properties": {
                "vector": {
                    "type": "knn_vector",
                    "dimension": dim
                },
                "chunk": {
                    "type": "text"
                },
                "doc_id": {
                    "type": "keyword"
                }
            }
        }
    }

    print(f"Creating index '{index_name}' with knn_vector mapping...")
    os_client.indices.create(index=index_name, body=body)
    print(f"Index '{index_name}' created.")


def main():
    # Step 1: Load envs 
    creds = json.loads(os.environ["PINECONE_API_KEY"])  # rename later, but fine for now
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

    # Step 5: Set up OpenSearch Serverless client and vector store
    region = "us-east-1"
    service = "aoss"

    endpoint = "https://wh3q028h23sdc9u2hsab.us-east-1.aoss.amazonaws.com"
    index_name = "test"
    dim = 1536  # text-embedding-3-small

    session = boto3.Session()
    credentials = session.get_credentials()
    awsauth = AWSV4SignerAuth(credentials, region, service)

    # Ensure index exists with proper knn_vector mapping
    ensure_index(endpoint, awsauth, index_name, dim, region)

    client = OpensearchVectorClient(
        endpoint,
        index_name,
        dim,
        embedding_field="vector",
        text_field="chunk",
        http_auth=awsauth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        timeout=60,
        max_retries=3,
        retry_on_timeout=True,
    )

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
    print("Ingestion complete — data stored in OpenSearch.")


if __name__ == "__main__":
    main()
