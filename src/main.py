import os, json, boto3
from requests_aws4auth import AWS4Auth
from opensearchpy import OpenSearch, RequestsHttpConnection
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_docling import DoclingLoader

def main():
    # Step 1: Get env variables
    bucket_name = os.environ["S3_BUCKET_NAME"]
    s3_key = os.environ["S3_OBJECT_KEY"]
    creds = json.loads(os.environ["PINECONE_API_KEY"])  # assumes JSON with OPENAI_API_KEY

    # Step 2: Create a presigned S3 URL for document
    s3 = boto3.client("s3", region_name="us-east-1")
    presigned_url = s3.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": bucket_name, "Key": s3_key},
        ExpiresIn=3600,
    )

    # Step 3: Read + Chunk document
    loader = DoclingLoader(file_path=presigned_url)
    chunks = loader.load()

    # Step 4: Connect to OpenSearch Service
    region = "us-east-1"
    service = "es"

    credentials = boto3.Session().get_credentials()
    awsauth = AWS4Auth(
        credentials.access_key,
        credentials.secret_key,
        region,
        service,
        session_token=credentials.token,
    )

    opensearch_host = "search-burrow-domain-mno6g2tea56zkyb6k2bhhsl7tm.us-east-1.es.amazonaws.com"

    vector_client = OpenSearch(
        hosts=[{"host": opensearch_host, "port": 443}],
        http_auth=awsauth,
        use_ssl=True,
        verify_certs=True,
        http_compress=True,
        connection_class=RequestsHttpConnection,
    )

    # Step 5: Create the index (k-NN enabled, lucene + L2)
    index_name = "burrow-index"
    index_body = {
        "settings": {
            "index": {
                "knn": True
            }
        },
        "mappings": {
            "properties": {
                "vector_field": {
                    "type": "knn_vector",
                    "dimension": 1536,
                    "method": {
                        "engine": "lucene",
                        "name": "hnsw",
                        "space_type": "l2",
                    },
                }
            }
        },
    }

    if not vector_client.indices.exists(index=index_name):
        vector_client.indices.create(index=index_name, body=index_body)

    # Step 6: Embeddings model (normalized => cosine via L2)
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=creds["OPENAI_API_KEY"],
    )

    # Step 7: Connect to OpenSearch vector index via LangChain
    opensearch_url = f"https://{opensearch_host}"

    vector_store = OpenSearchVectorSearch(
        opensearch_url=opensearch_url,
        index_name=index_name,
        embedding_function=embeddings,
        http_auth=awsauth,
        use_ssl=True,
        verify_certs=True,
        http_compress=True,
        connection_class=RequestsHttpConnection,
    )

    # Step 8: Load chunks into vector store
    texts = [doc.page_content for doc in chunks]
    metadatas = [doc.metadata for doc in chunks]

    vector_store.add_texts(
        texts=texts,
        metadatas=metadatas,
    )


if __name__ == "__main__":
    main()
