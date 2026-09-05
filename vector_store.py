from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

reader = PdfReader("data/notes.pdf")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = []
metadatas = []

# Process each page separately
for page_number, page in enumerate(reader.pages, start=1):

    text = page.extract_text()

    if text:
        page_chunks = splitter.split_text(text)

        for chunk in page_chunks:
            chunks.append(chunk)
            metadatas.append({
                "page": page_number
            })

print("Number of chunks:", len(chunks))

# Create embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks).tolist()

# Connect to ChromaDB
client = chromadb.PersistentClient(path="chroma_db")

# Rebuild collection
try:
    client.delete_collection("study_notes")
except:
    pass

collection = client.create_collection(
    name="study_notes"
)

# Store chunks + page numbers
collection.add(
    ids=[f"chunk_{i}" for i in range(len(chunks))],
    documents=chunks,
    embeddings=embeddings,
    metadatas=metadatas
)

print("Stored chunks:", collection.count())
print("Vector database updated with page numbers!")