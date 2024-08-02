from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

def setup_embeddings(model_name, device):
    model_kwargs = {"device": device}
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
    return embeddings

def setup_vectorstore(documents, embeddings, persist_directory):
    vectordb = Chroma.from_documents(documents=all_splits, embedding=embeddings, persist_directory=persist_directory)
    return vectordb

def setup_retriever(vectordb):
    return vectordb.as_retriever()