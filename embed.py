from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# 1. Read PDF
reader = PdfReader("data/notes.pdf")

all_text = ""

for page in reader.pages:
    text = page.extract_text()
    if text:
        all_text += text + "\n"

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_text(all_text)

print("Number of chunks:", len(chunks))

# 3. Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 4. Create embeddings
embeddings = model.encode(chunks)

print("Embedding shape:", embeddings.shape)