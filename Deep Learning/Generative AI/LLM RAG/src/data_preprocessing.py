from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_documents(file_path):
    loader = TextLoader(file_path, encoding="utf8")
    documents = loader.load()
    return documents

def split_text(documents, chunk_size=1000, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_splits = text_splitter.split_documents(documents)
    return all_splits