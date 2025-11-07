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

# Load environment variables from .env file
load_dotenv()

def main (file_path):
    connection_string = "http://host.docker.internal:3000/v1/api/bucketconfig"
    file_dir = os.path.dirname(file_path)
    folder_path_for_query = os.path.basename(file_dir) if file_dir else "wild_animals"  # Extract "wild_animals" from path

    try:
        response = requests.get(f"{connection_string}/{folder_path_for_query}")

        if response.status_code != 200:
            print(f"API request failed with status code: {response.status_code}")
            exit(1)  # Exit with error code 1

        print(response)
        config = response.json()
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
    md_file_path = f"../files/{folder_path_for_query}/{filename_without_extension}.md"

    # Code to create md file for parse file 
    with open(md_file_path, "w", encoding="utf-8") as f:
        f.write(parsed_md )

    # Load the parse.md file
    file_md = SimpleDirectoryReader(input_files=[md_file_path]).load_data()

    # Initialize connection to Pinecone
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

    # Get or create index (text-embedding-3-small has 1536 dimensions)
    # Create your index (can skip this step if your index already exists)
    try:
        pc.create_index(
            index_name,
            dimension=1536,  # text-embedding-3-small dimension
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    except Exception:
        # Index already exists, continue
        pass

    # Initialize your index
    pinecone_index = pc.Index(index_name)

    # Initialize VectorStore with namespace
    vector_store = PineconeVectorStore(
        pinecone_index=pinecone_index,
        namespace=namespace
    )

    # Create embedding model
    embed_model = OpenAIEmbedding(model="text-embedding-3-small")

    # Create ingestion pipeline with transformations
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap),  # measure by tokens
            embed_model,
        ],
        vector_store=vector_store,
    )

    # Ingest directly into vector db
    pipeline.run(documents=file_md)

# main("../files/wild_animals/lion.pdf")
main("../files/domestic_animals/cat.pdf")
# main("../files/lion.pdf")