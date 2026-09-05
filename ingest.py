from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

reader = PdfReader("data/notes.pdf")

all_text = ""

for page in reader.pages:
    text = page.extract_text()

    if text:
        all_text += text + "\n"

print("Total characters:", len(all_text))

# Create a text splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_text(all_text)

print("Number of chunks:", len(chunks))

print("\n--- First Chunk ---")
print(chunks[0])

print("\n--- Second Chunk ---")
print(chunks[1])