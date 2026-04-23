import os
import json
import re
from dotenv import load_dotenv

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# =====================
# ENV
# =====================
load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

PDF_PATH = "regulation.pdf"
JSON_PATH = "faculty_QA_dataset.json"
DB_PATH = "./academic_db"

# =====================
# LOAD DATA
# =====================
def load_docs():

    pdf_docs = PyPDFLoader(PDF_PATH).load()
    for d in pdf_docs:
        d.metadata = {"source": "pdf"}

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    def flat(x):
        if isinstance(x, dict):
            return " | ".join(f"{k}: {flat(v)}" for k, v in x.items())
        elif isinstance(x, list):
            return " | ".join(flat(i) for i in x)
        return str(x)

    json_docs = [
        Document(page_content=flat(i), metadata={"source": "json"})
        for i in (data if isinstance(data, list) else [data])
    ]

    return pdf_docs + json_docs

# =====================
# COURSE MAP (code ↔ name)
# =====================
def build_course_map(docs):

    course_map = {}

    pattern = r"(\d{2}-\d{2}-\d{5}).*?-\s*(.*?)\s*\("

    for doc in docs:
        matches = re.findall(pattern, doc.page_content)

        for code, name in matches:
            course_map[code] = name

    return course_map

# =====================
# VECTOR DB
# =====================
def build_db(docs):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH
    )

# =====================
# INIT
# =====================
docs = load_docs()
course_map = build_course_map(docs)
vectorstore = build_db(docs)
retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

# =====================
# PROMPT
# =====================
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a strict academic assistant.

RULES:
- Use ONLY context
- DO NOT guess
- If not found: "Not found in data"

Context:
{context}

Question:
{question}

Answer:
"""
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt}
)

# =====================
# FORMAT (IMPORTANT FIX)
# =====================
def format_answer(text):

    for code, name in course_map.items():
        if code in text:
            text = text.replace(code, f"{name} ({code})")

    return text

# =====================
# MAIN FUNCTION
# =====================
def ask_rag(q):

    res = qa.invoke({"query": q})["result"]

    return format_answer(res)