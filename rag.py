from sentence_transformers import SentenceTransformer
import chromadb
import requests

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to ChromaDB
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection("study_notes")

# Ask question
question = input("Ask your question: ")

# Convert question into embedding
query_embedding = model.encode([question]).tolist()

# Retrieve relevant chunks
results = collection.query(
    query_embeddings=query_embedding,
    n_results=3
)

# Show retrieved sources with page numbers
print("\n===== Retrieved Sources =====\n")

for i, document in enumerate(results["documents"][0], start=1):
    page = results["metadatas"][0][i - 1]["page"]

    print(f"--- Source {i} | Page {page} ---")
    print(document[:500])
    print()

# Combine retrieved chunks
context = "\n\n".join(results["documents"][0])

# Create prompt for Llama
prompt = f"""
You are a helpful study assistant.

Answer the question using ONLY the context provided below.

If the answer is not present in the context, say:
"I could not find this in the notes."

Context:
{context}

Question:
{question}

Answer:
"""

# Send prompt to Ollama
response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "llama3.2:3b",
        "prompt": prompt,
        "stream": False
    }
)

# Get Llama's answer
answer = response.json()["response"]

print("\n===== StudyMate Answer =====\n")
print(answer)