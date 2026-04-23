import os
import json

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

# =========================
# LOAD PDF
# =========================

PDF_PATH = "regulation.pdf"

loader = PyPDFLoader(PDF_PATH)
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(docs)

# =========================
# EMBEDDINGS
# =========================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# =========================
# GROQ LLM
# =========================

api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=api_key,
    temperature=0
)

# =========================
# JSON QA SYSTEM
# =========================

with open("faculty_QA_dataset.json", "r", encoding="utf-8") as f:
    qa_data = json.load(f)

def search_json(query):
    for item in qa_data:
        if query.lower() in item["question"].lower():
            return item["answer"]
    return None

# =========================
# MAIN FUNCTION
# =========================

def ask_rag(question):

    # 1. Greeting handling
    greetings = ["hi", "hello", "hey", "سلام", "ازيك"]
    if any(g in question.lower() for g in greetings):
        return "👋 Hello! Ask me anything from the PDF or dataset.", []

    # 2. JSON first
    json_answer = search_json(question)
    if json_answer:
        return json_answer, []

    # 3. PDF retrieval
    docs = retriever.get_relevant_documents(question)

    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
You are an academic assistant.

Answer ONLY from context.

Context:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    return response.content, docs
