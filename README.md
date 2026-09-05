\# 📚 StudyMate RAG



An AI-powered study notes assistant that uses Retrieval-Augmented Generation (RAG) to answer questions from uploaded PDF study notes.



\## 🚀 Features



\- 📄 Upload study notes in PDF format

\- 🔍 Semantic search using vector embeddings

\- 🧠 RAG-based question answering

\- 🤖 Local LLM inference using Ollama

\- 📖 Page-level source citations

\- 📝 Generate 5-question MCQ quizzes

\- 🎯 Interactive quiz with score calculation

\- 🔒 Answers are grounded only in the uploaded notes



\## 🛠️ Tech Stack



\- Python

\- Streamlit

\- Sentence Transformers

\- ChromaDB

\- LangChain Text Splitters

\- Ollama

\- Llama 3.2

\- PyPDF



\## 🏗️ RAG Architecture



PDF Notes  

↓  

Text Extraction  

↓  

Chunking  

↓  

Sentence Transformers  

↓  

Vector Embeddings  

↓  

ChromaDB  

↓  

Semantic Retrieval  

↓  

Relevant Context  

↓  

Llama 3.2  

↓  

Grounded Answer



\## ⚙️ How to Run



\### 1. Clone the repository



```bash

git clone https://github.com/srisaisumanth16-pixel/study-notes-rag.git

cd study-notes-rag2. Create virtual environment

python -m venv venv

3\. Install dependencies

pip install pypdf langchain-text-splitters sentence-transformers chromadb streamlit requests

4\. Start Ollama



Make sure Ollama is installed and run:



ollama run llama3.2:3b

5\. Run the application

streamlit run app.py

📌 Example

Question



What is Proof of Work?



Answer



StudyMate retrieves the relevant sections from the uploaded notes and generates a grounded answer using Llama 3.2.



🔮 Future Improvements

RAG evaluation using Hit@K and MRR

Better quiz generation

Multiple PDF support

Conversation history

Cloud deployment

Authentication

Improved citation system

👨‍💻 Author



Sri Sai Sumanth



GitHub: https://github.com/srisaisumanth16-pixel

