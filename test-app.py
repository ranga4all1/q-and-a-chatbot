# Import required libraries

import warnings
from pydantic import BaseModel

import boto3
import streamlit as st

# Using Titan embedding models to generate Embedding
from langchain.chains.retrieval import create_retrieval_chain

from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.llms.bedrock import Bedrock

# Data ingestion
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Vector Embeddings and Vector store
from langchain_community.vectorstores import FAISS

# LLM models
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain


# Suppress the pydantic warnings about protected namespaces
warnings.filterwarnings("ignore", message="Field .* has conflict with protected namespace.*")

# Suppress the LangChain deprecation warning
warnings.filterwarnings("ignore", category=DeprecationWarning, message="The class `BedrockEmbeddings` was deprecated.*")


# Bedrock clients
bedrock = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)


# Implement data ingestion
def data_ingestion():
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()
    # Character split works better with this pdf data set
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=10000
    )
    docs = text_splitter.split_documents(documents)
    return docs


# Vector embeddings and Vector store
def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(
        docs,
        bedrock_embeddings
    )
    vectorstore_faiss.save_local("faiss_index")

# vectorstore
docs = data_ingestion()
get_vector_store(docs)
faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)

# Define LLMs to use from aws bedrock
def get_titan_lite_llm():
    llm = Bedrock(
        model_id="amazon.titan-text-lite-v1",
        client=bedrock,
        model_kwargs={
            "temperature": 0.5,
            "maxTokenCount": 200,
        }
    )
    return llm

def get_mistral_llm():
    llm = Bedrock(
        model_id="mistral.mistral-7b-instruct-v0:2",
        client=bedrock,
        model_kwargs={
            "temperature": 0.5,
            "max_tokens": 200,
        }
    )
    return llm

def get_claude_llm():
    llm = Bedrock(
        model_id="anthropic.claude-v2:1",
        client=bedrock,
        model_kwargs={
            "temperature": 0.5,
            "max_tokens_to_sample": 200,
        }
    )
    return llm

def get_llama3_llm():
    llm = Bedrock(
        model_id="meta.llama3-1-8b-instruct-v1:0",
        client=bedrock,
        model_kwargs={
            "temperature": 0.5,
            "max_gen_len": 256,
            # "top_p": 0.5,
        }
    )
    return llm


# create the prompt template
prompt_template = """
    Human: use the following pieces of context to provide a concise answer to the question at the end
    but at least summarize with 150 words with detailed explanation. If you don't know the answer, just say that you don't know, don't try to make up an answer. Do not repeat the answer or explanation.
    <context>
    {context}
    </context
    Question: {question}

    Assistant:
"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)


def get_response_llm(llm, vectorstore_faiss, query):
    question_answer_chain = create_stuff_documents_chain(llm, PROMPT)
    retriever = vectorstore_faiss.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    )
    qa = create_retrieval_chain(retriever, question_answer_chain)
    answer = qa.invoke({"input": query, "question": query})
    return answer['answer']


llms = [get_llama3_llm, get_titan_lite_llm, get_mistral_llm, get_claude_llm]

questions = [

    "Where is Ranga Hande currently located?",
    # "What is Ranga Hande's profession?",
    # "What projects has Ranga Hande worked on?",
    # "Where did Ranga Hande study?",
    # "What are Ranga Hande's areas of expertise?",
    # "Has Ranga Hande published any research papers or articles on 'Medium' or 'LinkedIn'?"
]

for user_question in questions:
    print(f"Question: {user_question}")
    print("-" * 50)

    for llm_func in llms:
        print(f"Using LLM function: {llm_func.__name__}")  # Print the function name
        llm = llm_func()
        result = get_response_llm(llm, faiss_index, user_question)
        print("Result:")
        print(result)
        print("\n" + "-"*50 + "\n")  # Add a separator between results

    print("\n" + "="*50 + "\n")  # Add a bigger separator between questions
