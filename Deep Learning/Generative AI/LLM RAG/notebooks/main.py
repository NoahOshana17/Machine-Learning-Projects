import os
import sys
sys.path.insert(0, os.path.abspath('../src'))

from src.model import check_cuda, download_model, setup_model, setup_pipeline, test_model
from src.data_processing import load_documents, split_text
from src.retrieval import setup_embeddings, setup_vectorstore, setup_retriever
from src.pipeline import setup_llm, setup_retrieval_qa, test_rag
from src.config import EMBEDDINGS_MODEL_NAME, CHROMA_DB_PATH, DOCUMENT_FILE_PATH

# Check CUDA availability
cuda_available = check_cuda()
print(f"CUDA available: {cuda_available}")

# Download model
model_name = "metaresearch/llama-2/pyTorch/7b-chat-hf"
model_path = download_model(model_name)

# Setup model and tokenizer
model_id = model_path
device = f'cuda:{cuda.current_device()}' if cuda_available else 'cpu'
model, tokenizer = setup_model(model_id, device)

# Setup query pipeline
query_pipeline = setup_pipeline(model, tokenizer)

# Test the model
test_prompt = "Please explain what is the State of the Union address. Give just a definition. Keep it in 100 words."
test_model(tokenizer, query_pipeline, test_prompt)

# Setup LLM
llm = setup_llm(query_pipeline)

# Load documents
documents = load_documents(DOCUMENT_FILE_PATH)

# Split text
all_splits = split_text(documents)

# Setup embeddings
embeddings = setup_embeddings(EMBEDDINGS_MODEL_NAME, device)

# Setup vector store
vectordb = setup_vectorstore(all_splits, embeddings, persist_directory=CHROMA_DB_PATH)

# Setup retriever
retriever = setup_retriever(vectordb)

# Setup retrieval QA
qa = setup_retrieval_qa(llm, retriever)

# Test RAG pipeline
query = "What were the main topics in the State of the Union in 2023? Summarize. Keep it under 200 words."
test_rag(qa, query)