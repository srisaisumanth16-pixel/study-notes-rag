import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
import requests


# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="StudyMate",
    page_icon="📚"
)

st.title("📚 StudyMate")
st.write("AI Study Notes Assistant")


# ==========================================
# LOAD EMBEDDING MODEL
# ==========================================

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


model = load_model()


# ==========================================
# CHROMA DATABASE
# ==========================================

client = chromadb.PersistentClient(
    path="chroma_db"
)


# ==========================================
# GET COLLECTION
# ==========================================

def get_collection():

    if "collection_name" in st.session_state:
        return client.get_collection(
            st.session_state["collection_name"]
        )

    return client.get_collection(
        "study_notes"
    )


# ==========================================
# PDF UPLOAD
# ==========================================

uploaded_file = st.file_uploader(
    "📤 Upload your study notes (PDF)",
    type=["pdf"]
)


if uploaded_file is not None:

    if st.button("Process PDF"):

        reader = PdfReader(uploaded_file)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = []
        metadatas = []

        # --------------------------------------
        # READ PDF PAGE BY PAGE
        # --------------------------------------

        for page_number, page in enumerate(
            reader.pages,
            start=1
        ):

            text = page.extract_text()

            if not text:
                continue

            text = text.strip()

            # Ignore almost empty pages
            if len(text) < 50:
                continue

            page_chunks = splitter.split_text(
                text
            )

            for chunk in page_chunks:

                chunk = chunk.strip()

                # Ignore useless chunks
                if len(chunk) < 50:
                    continue

                chunks.append(chunk)

                metadatas.append({
                    "page": page_number
                })

        st.write(
            f"📄 Pages: {len(reader.pages)}"
        )

        st.write(
            f"🧩 Useful chunks: {len(chunks)}"
        )

        if len(chunks) == 0:

            st.error(
                "❌ No readable text was found in the PDF."
            )

        else:

            # --------------------------------------
            # CREATE EMBEDDINGS
            # --------------------------------------

            embeddings = model.encode(
                chunks
            ).tolist()

            # --------------------------------------
            # DELETE OLD COLLECTION
            # --------------------------------------

            try:

                client.delete_collection(
                    "uploaded_notes"
                )

            except:

                pass

            # --------------------------------------
            # CREATE COLLECTION
            # --------------------------------------

            collection = client.create_collection(
                name="uploaded_notes"
            )

            collection.add(
                ids=[
                    f"uploaded_chunk_{i}"
                    for i in range(len(chunks))
                ],
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadatas
            )

            st.session_state[
                "collection_name"
            ] = "uploaded_notes"

            st.success(
                "✅ PDF processed successfully!"
            )


# ==========================================
# ASK STUDYMATE
# ==========================================

st.divider()

st.subheader("💬 Ask StudyMate")

question = st.text_input(
    "Ask a question from your notes:"
)


if st.button("Ask StudyMate") and question:

    collection = get_collection()

    # --------------------------------------
    # CREATE QUESTION EMBEDDING
    # --------------------------------------

    query_embedding = model.encode(
        [question]
    ).tolist()

    # --------------------------------------
    # RETRIEVE TOP 3 CHUNKS
    # --------------------------------------

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3
    )

    documents = results["documents"][0]

    # --------------------------------------
    # CREATE CONTEXT
    # --------------------------------------

    context = "\n\n".join(
        documents
    )

    # --------------------------------------
    # RAG PROMPT
    # --------------------------------------

    prompt = f"""
You are StudyMate, an AI study assistant.

Answer the question using ONLY the study notes
provided below.

Do NOT use outside knowledge.

If the answer is not directly available in the
study notes, respond EXACTLY:

I could not find this in the notes.

Study Notes:
{context}

Question:
{question}

Answer:
"""

    # --------------------------------------
    # CALL OLLAMA
    # --------------------------------------

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.2:3b",
            "prompt": prompt,
            "stream": False
        }
    )

    answer = response.json()["response"].strip()

    # --------------------------------------
    # CHECK NOT FOUND
    # --------------------------------------

    not_found = (
        "i could not find this in the notes"
        in answer.lower()
        or
        "i couldn't find this in the notes"
        in answer.lower()
    )

    # --------------------------------------
    # ANSWER NOT FOUND
    # --------------------------------------

    if not_found:

        st.subheader(
            "🤖 StudyMate Answer"
        )

        st.info(
            "I could not find this in the notes."
        )

        # IMPORTANT:
        # No sources are displayed.

    # --------------------------------------
    # ANSWER FOUND
    # --------------------------------------

    else:

        st.subheader(
            "🤖 StudyMate Answer"
        )

        st.write(answer)

        # --------------------------------------
        # SHOW SOURCES
        # --------------------------------------

        st.subheader(
            "📖 Sources"
        )

        for i, document in enumerate(
            documents,
            start=1
        ):

            page = results[
                "metadatas"
            ][0][i - 1]["page"]

            with st.expander(
                f"Source {i} — Page {page}"
            ):

                st.write(document)


# ==========================================
# QUIZ MODE
# ==========================================

st.divider()

st.subheader("📝 Quiz Mode")

quiz_topic = st.text_input(
    "Enter a topic for the quiz:",
    placeholder="Example: Proof of Work"
)


if st.button("Generate MCQs"):

    if not quiz_topic.strip():

        st.warning(
            "⚠️ Please enter a quiz topic."
        )

    else:

        collection = get_collection()

        # --------------------------------------
        # SEARCH VECTOR DATABASE
        # --------------------------------------

        topic_embedding = model.encode(
            [quiz_topic]
        ).tolist()

        search_results = collection.query(
            query_embeddings=topic_embedding,
            n_results=5
        )

        documents = search_results[
            "documents"
        ][0]

        distances = search_results[
            "distances"
        ][0]

        # --------------------------------------
        # CREATE QUIZ CONTEXT
        # --------------------------------------

        quiz_context = "\n\n".join(
            documents
        )

        # --------------------------------------
        # TOPIC KEYWORD CHECK
        # --------------------------------------

        topic_words = [
            word.lower().strip(
                ".,!?;:()[]{}"
            )
            for word in quiz_topic.split()
            if len(
                word.strip(
                    ".,!?;:()[]{}"
                )
            ) > 2
        ]

        context_lower = quiz_context.lower()

        keyword_matches = 0

        for word in topic_words:

            if word in context_lower:

                keyword_matches += 1

        # --------------------------------------
        # DETERMINE WHETHER TOPIC EXISTS
        # --------------------------------------

        topic_found = False

        # Exact phrase
        if quiz_topic.lower() in context_lower:

            topic_found = True

        # All important words found
        elif (
            len(topic_words) > 0
            and keyword_matches
            == len(topic_words)
        ):

            topic_found = True

        # Semantic similarity
        elif len(distances) > 0:

            best_distance = min(distances)

            if best_distance < 1.2:

                topic_found = True

        # ======================================
        # TOPIC NOT FOUND
        # ======================================

        if not topic_found:

            st.error(
                "❌ I couldn't find this topic in the notes."
            )

            st.info(
                "Please enter a topic that is covered "
                "in your uploaded PDF."
            )

            st.session_state.pop(
                "quiz_questions",
                None
            )

        # ======================================
        # TOPIC FOUND
        # ======================================

        else:

            # ----------------------------------
            # GENERATE MCQs
            # ----------------------------------

            quiz_prompt = f"""
You are an exam quiz generator.

Create EXACTLY 5 multiple-choice questions
about:

{quiz_topic}

Use ONLY the study notes below.

Do NOT use outside knowledge.
Do NOT invent information.

Every question must be directly related
to the requested topic.

Use EXACTLY this format:

QUESTION: question text
A: option A
B: option B
C: option C
D: option D
ANSWER: A

Repeat exactly 5 times.

Rules:
- Exactly 5 questions.
- Exactly 4 options.
- Only one correct answer.
- ANSWER must be A, B, C, or D.
- Every question must be about {quiz_topic}.
- Every answer must be supported by the notes.
- No explanations.

Study Notes:

{quiz_context}
"""

            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3.2:3b",
                    "prompt": quiz_prompt,
                    "stream": False
                }
            )

            quiz_text = response.json()[
                "response"
            ]

            # ----------------------------------
            # PARSE QUESTIONS
            # ----------------------------------

            raw_questions = quiz_text.split(
                "QUESTION:"
            )

            questions = []

            for raw_question in raw_questions[1:]:

                lines = [
                    line.strip()
                    for line in raw_question.splitlines()
                    if line.strip()
                ]

                if len(lines) < 6:
                    continue

                question_text = lines[0]

                options = {}

                correct_answer = None

                for line in lines:

                    if line.startswith("A:"):

                        options["A"] = (
                            line[2:].strip()
                        )

                    elif line.startswith("B:"):

                        options["B"] = (
                            line[2:].strip()
                        )

                    elif line.startswith("C:"):

                        options["C"] = (
                            line[2:].strip()
                        )

                    elif line.startswith("D:"):

                        options["D"] = (
                            line[2:].strip()
                        )

                    elif line.startswith(
                        "ANSWER:"
                    ):

                        correct_answer = (
                            line.split(
                                "ANSWER:"
                            )[1]
                            .strip()
                            .upper()
                        )

                # Accept only complete questions
                if (
                    len(options) == 4
                    and correct_answer in options
                    and question_text
                ):

                    questions.append({
                        "question": question_text,
                        "options": options,
                        "answer": correct_answer
                    })

            # ----------------------------------
            # VALIDATE 5 QUESTIONS
            # ----------------------------------

            if len(questions) < 5:

                st.error(
                    f"⚠️ Only {len(questions)} valid "
                    "questions were generated."
                )

                st.info(
                    "Please click Generate MCQs again."
                )

                st.session_state.pop(
                    "quiz_questions",
                    None
                )

            else:

                # Keep exactly 5
                questions = questions[:5]

                st.session_state[
                    "quiz_questions"
                ] = questions

                # Clear old answers
                for i in range(1, 6):

                    st.session_state.pop(
                        f"quiz_answer_{i}",
                        None
                    )

                st.success(
                    "✅ 5 questions generated!"
                )


# ==========================================
# DISPLAY QUIZ
# ==========================================

if "quiz_questions" in st.session_state:

    questions = st.session_state[
        "quiz_questions"
    ]

    st.subheader(
        "📚 Practice Quiz"
    )

    for i, q in enumerate(
        questions,
        start=1
    ):

        st.write(
            f"### Question {i}"
        )

        st.write(
            f"**{q['question']}**"
        )

        option_list = [
            "Select an option",
            "A",
            "B",
            "C",
            "D"
        ]

        st.radio(
            "Choose your answer:",
            option_list,
            format_func=lambda x,
            opts=q["options"]: (
                "Please select an answer"
                if x == "Select an option"
                else f"{x}: {opts[x]}"
            ),
            key=f"quiz_answer_{i}"
        )

        st.divider()

    # ======================================
    # SUBMIT QUIZ
    # ======================================

    if st.button("🎯 Submit Quiz"):

        score = 0
        unanswered = 0

        total_questions = len(
            questions
        )

        for i, q in enumerate(
            questions,
            start=1
        ):

            user_answer = st.session_state.get(
                f"quiz_answer_{i}",
                "Select an option"
            )

            correct_answer = q[
                "answer"
            ]

            if user_answer == "Select an option":

                unanswered += 1

                st.warning(
                    f"Question {i}: ⚠️ Not answered"
                )

            elif user_answer == correct_answer:

                score += 1

                st.success(
                    f"Question {i}: ✅ Correct"
                )

            else:

                st.error(
                    f"Question {i}: ❌ Wrong "
                    f"(Correct answer: "
                    f"{correct_answer})"
                )

        # ======================================
        # FINAL RESULT
        # ======================================

        percentage = (
            score / total_questions
        ) * 100

        st.subheader(
            "🏆 Quiz Result"
        )

        st.success(
            f"Final Score: "
            f"{score}/{total_questions}"
        )

        st.info(
            f"Percentage: {percentage:.0f}%"
        )

        if unanswered > 0:

            st.warning(
                f"Unanswered: {unanswered}"
            )