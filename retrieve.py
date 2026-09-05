from sentence_transformers import SentenceTransformer
import chromadb

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to ChromaDB
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection("study_notes")

# User question
question = "What is Proof of Work?"

# Convert question into embedding
query_embedding = model.encode([question]).tolist()

# Search similar chunks
results = collection.query(
    query_embeddings=query_embedding,
    n_results=3
)

# Display results
print("\nTop relevant chunks:\n")

for i, document in enumerate(results["documents"][0], start=1):
    print(f"--- Result {i} ---")
    print(document)
    print()