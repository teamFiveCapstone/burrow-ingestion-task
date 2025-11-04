from docling.document_converter import DocumentConverter
from chonkie import SentenceChunker
from pathlib import Path




# source = "../files/lion.pdf"  # document per local path or URL
# converter = DocumentConverter()
# result = converter.convert(source)

# lionPDF_md = result.document.export_to_markdown() #  pdf to markdown 


# # code to create md file for output 
# with open("../files/lion.md", "w", encoding="utf-8") as f:
#     f.write(lionPDF_md )



#chunking with chonkie (sentence chunker )

# Basic initialization with default parameters
chunker = SentenceChunker(
    tokenizer="character",     # Default tokenizer (or use "gpt2", etc.)
    chunk_size=800,           # Maximum tokens per chunk
    chunk_overlap=50,         # Overlap between chunks
    min_sentences_per_chunk=1  # Minimum sentences in each chunk
)


file_path = Path("/files/lion.md")
text = file_path.read_text(encoding="utf-8")

chunks = chunker.chunk(text)

# Write chunks to markdown file
with open("/files/chunked_lion.md", "w", encoding="utf-8") as f:
    chunk_lines = []
    for index, chunk in enumerate(chunks, start=1):
        chunk_text = chunk.text if hasattr(chunk, 'text') else str(chunk)
        chunk_lines.append(f'## Chunk {index}\n\n{chunk_text}\n\n')
    f.write(''.join(chunk_lines))


# print("Markdown saved to lion.md")
