from fileinput import filename
from docling.document_converter import DocumentConverter
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.ingestion import IngestionPipeline
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os
import requests
import boto3
import tempfile
import shutil
import json

# Load environment variables from .env file
load_dotenv()

def download_from_s3(bucket_name, s3_key):
    """Download file from S3 to temporary directory and return local path"""
    s3_client = boto3.client('s3')

    # Extract filename from S3 key
    filename = os.path.basename(s3_key)

    # Create temporary file
    temp_dir = tempfile.mkdtemp()
    local_path = os.path.join(temp_dir, filename)

    # Download file from S3
    s3_client.download_file(bucket_name, s3_key, local_path)

    return local_path

def main(bucket_name, s3_key):
    print(f'Whats my os environ bro: {os.environ["PINECONE_API_KEY"]}')
    data = json.loads(os.environ["PINECONE_API_KEY"])
    print(f'Pinecone key: {data["PINECONE_API_KEY"]}')
    print(data)

    # Download file from S3 to temporary location
    file_path = download_from_s3(bucket_name, s3_key)

    connection_string = "http://host.docker.internal:3000/v1/api/bucketconfig"
    # Extract folder name from S3 key instead of local file path
    s3_dir = os.path.dirname(s3_key)
    folder_path_for_query = os.path.basename(s3_dir) if s3_dir else "wild_animals"

    try:
        # response = requests.get(f"{connection_string}/{folder_path_for_query}")
        response = 200
        # response.status_code
        if response!= 200:
            print(f"API request failed with status code: {response.status_code}")
            exit(1)  # Exit with error code 1

        print(response)
        # config = response.json()
        config = {'_id': '690cf4920809672de613c84a', 'chunk_overlap': '20', 'chunk_size': '500', 'chunk_strategy': 'sentence', 'metadata_extraction': 'default', 'namespace': 'lion', 'folder_path': 'wild-cats-pipeline', 'index_name': 'lion'}

    except Exception as e:
      print(f"Error fetching config from API: {e}")
      exit(1)  # Exit with error code 1
    
    chunk_size = config.get('chunk_size')
    chunk_overlap = config.get('chunk_overlap')
    chunk_strategy = config.get('chunk_strategy')
    namespace = config.get('namespace')
    index_name = config.get('index_name')
    metadata_extraction = config.get("metadata_extraction")
    print(f"Using configs from API response: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}, namespace={namespace}, index_name={index_name}")
   
    # Parsing doc with Docling ****************************************************
    source = file_path # document per local path or URL
    converter = DocumentConverter()
    result = converter.convert(source)
    parsed_md = result.document.export_to_markdown() #  pdf to markdown 
    
    filename_with_extension = os.path.basename(file_path)
    filename_without_extension = os.path.splitext(filename_with_extension)[0]

    # Create temporary md file
    temp_md_dir = tempfile.mkdtemp()
    md_file_path = os.path.join(temp_md_dir, f"{filename_without_extension}.md")

    # Code to create md file for parse file
    with open(md_file_path, "w", encoding="utf-8") as f:
        f.write(parsed_md )

    # Load the parse.md file
    file_md = SimpleDirectoryReader(input_files=[md_file_path]).load_data()

    # Initialize connection to Pinecone
    pc = Pinecone(api_key=data["PINECONE_API_KEY"])
    
    print("successfully created pinecone instance")

    # Get or create index (text-embedding-3-small has 1536 dimensions)
    # Create your index (can skip this step if your index already exists)
    try:
        pc.create_index(
            index_name,
            dimension=1536,  # text-embedding-3-small dimension
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    except Exception:
        print("index already exists, skipping")
        pass

    pinecone_index = pc.Index(index_name)

    vector_store = PineconeVectorStore(
        pinecone_index=pinecone_index,
        namespace=namespace
    )

    embed_model = OpenAIEmbedding(
        model="text-embedding-3-small",
        api_key=data["OPENAI_API_KEY"]
        )

    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap),  # measure by tokens
            embed_model,
        ],
        vector_store=vector_store,
    )

    # Ingest directly into vector db
    pipeline.run(documents=file_md)

    # Cleanup temporary files
    temp_pdf_dir = os.path.dirname(file_path)
    temp_md_dir = os.path.dirname(md_file_path)

    try:
        shutil.rmtree(temp_pdf_dir)
        shutil.rmtree(temp_md_dir)
    except Exception as e:
        print(f"Warning: Could not clean up temporary files: {e}")

# Example usage:
bucket_name = os.environ["S3_BUCKET_NAME"]
s3_key = os.environ["S3_OBJECT_KEY"]
print(f"bucket_name: {bucket_name}")
print(f"s3_key: {s3_key}")
#bucket_name = "wild-cats-pipeline"
#s3_key = "wild-cats-pipeline/African-lion.pdf"
print(f"Trying to download: s3://{bucket_name}/{s3_key}")

# List bucket contents to debug
s3_client = boto3.client('s3')
try:
    response = s3_client.list_objects_v2(Bucket=bucket_name)
    print(f"Files in bucket '{bucket_name}':")
    if 'Contents' in response:
        for obj in response['Contents']:
            print(f"  - {obj['Key']}")
    else:
        print("  (bucket is empty)")
except Exception as e:
    print(f"Error listing bucket: {e}")

main(bucket_name, s3_key) 
# bucket_name: wild-cats-pipeline/
# s3_key: wild-cats-pipeline/African-lion.pdf