# 📘 RAG Application using LangChain, OpenAI API, and FAISS

This tutorial walks you through building a Retrieval-Augmented Generation (RAG) application using LangChain, OpenAI, and FAISS.

---

## 🔧 1. Setup and Document Loading

```python
# Install necessary packages
!pip install langchain openai tiktoken rapidocr-onnxruntime

# Load environment and libraries
import os
import dotenv
from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS
import requests

dotenv.load_dotenv()  # Load environment variables

# Load OpenAI API Key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Load document
with open("state_of_the_union.txt", "r", encoding="utf8") as f:
    data = f.read()

loader = TextLoader('state_of_the_union.txt', encoding="utf8")
document = loader.load()
print(document[0].page_content)
```

---

## ✂️ 2. Text Chunking

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Split the document into manageable chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
text_chunks = text_splitter.split_documents(document)
print(text_chunks[3].page_content)
```

---

## 🔍 3. Embedding and Vector Store with FAISS

```python
# Generate embeddings and create FAISS vectorstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(text_chunks, embeddings)
retriever = vectorstore.as_retriever()
```

---

## 🧠 4. RAG Pipeline with OpenAI

```python
from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# Define the RAG prompt
template = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use ten sentences maximum and keep the answer concise.
Question: {question}
Context: {context}
Answer:"""

prompt = ChatPromptTemplate.from_template(template)
output_parser = StrOutputParser()
llm_model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name="gpt-3.5-turbo")

# Create RAG pipeline
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm_model
    | output_parser
)

# Run example queries
rag_chain.invoke("How is the United States supporting Ukraine economically and militarily?")
rag_chain.invoke("What action is the U.S. taking to address rising gas prices?")
```

---
