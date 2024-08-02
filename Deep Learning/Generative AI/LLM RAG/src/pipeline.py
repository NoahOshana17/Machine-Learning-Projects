from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from time import time

def setup_llm(pipeline):
    return HuggingFacePipeline(pipeline=query_pipeline)

def setup_retrieval_qa(llm, retriever):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        verbose=True
    )
    return qa

def test_rag(qa, query):
    print(f"Query: {query}\n")
    time_1 = time()
    result = qa.run(query)
    time_2 = time()
    print(f"Inference time: {round(time_2 - time_1, 3)} sec.")
    print("\nResult: ", result)