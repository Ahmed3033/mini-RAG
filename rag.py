import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.documents import Document

# =====================
# LOAD DATA
# =====================

PDF_PATH = "regulation.pdf"
JSON_PATH = "faculty_QA_dataset.json"

loader = PyPDFLoader(PDF_PATH)
pdf_docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

pdf_chunks = splitter.split_documents(pdf_docs)

# =====================
# EMBEDDINGS
# =====================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma.from_documents(
    documents=pdf_chunks,
    embedding=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# =====================
# LLM (GROQ)
# =====================

api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=api_key,
    temperature=0
)

# =====================
# SIMPLE JSON QA
# =====================

import json

with open(JSON_PATH, "r", encoding="utf-8") as f:
    qa_data = json.load(f)

def search_json(query):
    for item in qa_data:
        if query.lower() in item["question"].lower():
            return item["answer"]
    return None

# =====================
# MAIN FUNCTION
# =====================

def ask_rag(question):

    # 1. check JSON first
    json_answer = search_json(question)
    if json_answer:
        return json_answer, []

    # 2. retrieve from PDF
    docs = retriever.get_relevant_documents(question)

    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
You are an academic assistant.
Answer ONLY from the context.

Context:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    return response.content, docs
