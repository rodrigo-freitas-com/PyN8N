import os
import shutil

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma


from config import RAG_FILES_DIR, VECTOR_STORE_PATH


def load_documents():
    docs = []
    processed_dir = os.path.join(RAG_FILES_DIR, "processed")
    os.makedirs(processed_dir, exist_ok=True)

    if not os.path.exists(RAG_FILES_DIR):
        return docs

    files = [
        os.path.join(RAG_FILES_DIR, f)
        for f in os.listdir(RAG_FILES_DIR)
        if f.endswith(".pdf") or f.endswith(".txt")
    ]

    for file in files:
        try:
            loader = PyPDFLoader(file) if file.endswith(".pdf") else TextLoader(file)
            docs.extend(loader.load())

            dest_path = os.path.join(processed_dir, os.path.basename(file))
            shutil.move(file, dest_path)

        except Exception as e:
            print(f"Erro ao processar {file}: {e}")

    return docs


def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model="text-embedding-004",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )


def get_vectorstore():
    embeddings = get_embeddings()
    docs = load_documents()

    if docs:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )

        splits = text_splitter.split_documents(docs)

        return Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=VECTOR_STORE_PATH,
        )

    return Chroma(
        embedding_function=embeddings,
        persist_directory=VECTOR_STORE_PATH,
    )
