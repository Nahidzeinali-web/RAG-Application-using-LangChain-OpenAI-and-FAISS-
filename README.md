# 📘 RAG Application using LangChain, OpenAI API, and FAISS

This tutorial walks you through building a Retrieval-Augmented Generation (RAG) application using LangChain, OpenAI, and FAISS.

---

## 🔧 1. Setup and Document Loading

```python
# Install necessary packages
!pip install langchain openai tiktoken rapidocr-onnxruntime

# Import required modules
import os  # For environment and path operations
import dotenv  # To load environment variables from a .env file
from langchain.document_loaders import TextLoader  # Load text files into LangChain
from langchain.vectorstores import FAISS  # FAISS vector store for similarity search
import requests  # To make HTTP requests (not used but commonly included)

# Load environment variables from .env file
dotenv.load_dotenv()

# Get your OpenAI API key from environment
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Manually open and read the text file
with open("state_of_the_union.txt", "r", encoding="utf8") as f:
    data = f.read()

# Load document using LangChain's TextLoader
loader = TextLoader('state_of_the_union.txt', encoding="utf8")
document = loader.load()

# Print the content of the first document chunk
print(document[0].page_content)
```

---

## ✂️ 2. Text Chunking

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter  # For text chunking

# Split document into smaller chunks for embedding and retrieval
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
text_chunks = text_splitter.split_documents(document)

# Print the fourth chunk as an example
print(text_chunks[3].page_content)
```

---

## 🔍 3. Embedding and Vector Store with FAISS

```python
from langchain.embeddings import OpenAIEmbeddings  # Embedding model
from langchain.prompts import ChatPromptTemplate  # Template for chat-based prompts
from langchain.vectorstores import FAISS  # FAISS for vector search
from dotenv import load_dotenv  # To reload .env file if needed

# Load environment variables again (optional if already loaded above)
load_dotenv()

# Initialize OpenAI embedding model
embeddings = OpenAIEmbeddings()

# Create a FAISS vector store from text chunks
vectorstore = FAISS.from_documents(text_chunks, embeddings)

# Convert vectorstore into a retriever for searching
retriever = vectorstore.as_retriever()
```

---

## 🧠 4. RAG Pipeline with OpenAI

```python
from langchain.chat_models import ChatOpenAI  # Chat model from OpenAI
from langchain.schema.runnable import RunnablePassthrough  # For passing the query directly
from langchain.schema.output_parser import StrOutputParser  # To parse output as string

# Template for retrieving answers using context
template = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use ten sentences maximum and keep the answer concise.
Question: {question}
Context: {context}
Answer:"""

# Create a prompt object from the template
prompt = ChatPromptTemplate.from_template(template)

# Parse model output as plain string
output_parser = StrOutputParser()

# Initialize OpenAI chat model
llm_model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name="gpt-3.5-turbo")

# Create the full RAG chain pipeline
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}  # Context from retriever, question directly passed
    | prompt  # Format into prompt
    | llm_model  # Generate response using LLM
    | output_parser  # Convert output to string
)

# Example query 1
rag_chain.invoke("How is the United States supporting Ukraine economically and militarily?")

# Example query 2
rag_chain.invoke("What action is the U.S. taking to address rising gas prices?")
```

---


## 🙏 Acknowledgment
This implementation is inspired by tutorials from **Sunny Savita's YouTube channel**:
- [YouTube channel](https://youtu.be/y_act32Gjbc?si=5kSKvRwkQoVnswAh)

# Run example queries
rag_chain.invoke("How is the United States supporting Ukraine economically and militarily?")
rag_chain.invoke("What action is the U.S. taking to address rising gas prices?")
```

---
