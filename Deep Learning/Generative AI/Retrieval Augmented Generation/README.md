# Retrieval Augmented Generation with Meta's LLaMA 2

## Overview
This project demonstrates the development and implementation of a Retrieval Augmented Generation
(RAG) model using Meta's LLaMA 2 7B model. The goal is to enhance the language model's 
performance by integrating retrieval capabilities, leveraging a combination of 
state-of-the-art libraries and tools for efficient and accurate text generation.

## Problem Statement
Standard language models often generate text based on their training data alone, 
which can lead to less accurate or less relevant responses. By incorporating a retrieval 
mechanism, this project aims to improve the relevance and accuracy of the generated text, 
making it more reliable for practical applications such as summarization and question answering.

## Dataset
The project utilizes text data from:
- The 2023 State of the Union address.
- Personal information about the project's author, Noah Oshana.

## Models Used
- **LLaMA 2 7B**: Meta's pre-trained LLaMA 2 model, optimized with quantization techniques to reduce GPU memory usage.

## Technologies Used
- Python
- PyTorch
- Hugging Face Transformers
- ChromaDB
- LangChain
- Hugging Face Datasets
- bitsandbytes (for quantization)

## Learning Objectives
The primary objectives of this project include:
- Understanding and implementing a Retrieval Augmented Generation (RAG) model.
- Utilizing quantization techniques to optimize large language models for efficient GPU memory usage.
- Applying advanced text retrieval mechanisms to enhance language model outputs.

## Approach and Methodology
The project follows these steps:
1. **Model Setup**: Loading and configuring the LLaMA 2 7B model with quantization to handle large models efficiently.
2. **Data Preparation**: Loading, splitting, and embedding text data for retrieval purposes.
3. **Pipeline Configuration**: Setting up the text generation pipeline using Hugging Face Transformers and integrating it with LangChain.
4. **Retrieval Mechanism**: Building a vector database using ChromaDB to store document embeddings and configuring a retriever for fetching relevant document chunks.
5. **Query Processing**: Implementing functions to test the RAG model with different queries and evaluating its performance.

## Model Performance    
The performance of the RAG model is evaluated based on its ability to generate accurate 
and relevant responses to queries. Sample queries include summarizing the State of the 
Union address and providing personal information about the author.

## Impact
The integration of retrieval mechanisms with large language models can significantly 
enhance the relevance and accuracy of generated text, benefiting various applications 
such as information retrieval, summarization, and question answering.

## Future Enhancements
Future enhancements and experiments to improve model performance and application include:
- Fine-tuning the retrieval mechanism with more diverse and extensive datasets.
- Exploring additional quantization techniques to further optimize model efficiency.
- Integrating real-time retrieval and generation capabilities for dynamic applications.

## Example Queries
- **State of the Union Summary**:
  ```python
  query = "What were the main topics in the State of the Union in 2023? Summarize. Keep it under 200 words."
  test_rag(qa, query)

  '''python
  Question: What were the main topics in the State of the Union in 2023? Summarize. Keep it under 200 words.
  Helpful Answer: The State of the Union in 2023 focused on several key topics, including the nation's economic strength, the competition with China, and the need to come together as a nation to face the challenges ahead.   The President emphasized the importance of American innovation, industries, and military modernization to ensure the country's safety and stability. The President also highlighted the nation's resilience and optimism,     urging Americans to see each other as fellow citizens and to work together to overcome the challenges facing the country.
