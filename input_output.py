import fitz  # PyMuPDF
from pypdf import PdfReader
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import io
import groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
import os
from groq import Groq
from langchain.prompts import ChatPromptTemplate

PROMPT_TEMPLATE = """
Answer the user query using only the relevant information provided to you in this prompt.

Relevant context from documents:
{relevant}

user query: {query}
"""

messages = [{"role": "system", "content": "You are a helpful chatbot"}]

def get_similar(query, db):
    result = db.similarity_search_with_relevance_scores(query=query, k=10)
    return result

def ask_ques(user_query, messages, db):
    template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = template.format(
        relevant=get_similar(user_query, db),
        query=user_query
    )

    client = Groq(
        api_key="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",
    )

    messages.append({"role": "user", "content": prompt})

    chat_completion = client.chat.completions.create(
        messages=messages,
        model="llama3-8b-8192",
    )

    messages.pop()
    messages.append({"role": "user", "content": user_query})
    messages.append({"role": "assistant", "content": chat_completion.choices[0].message.content})

    print(f"\n\nResponse: {chat_completion.choices[0].message.content}")

def get_stuff(text, image_text, user_query, messages):
    text = text + image_text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    )

    chunks = text_splitter.split_text(text)

    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"trust_remote_code":True}) 

    db = Chroma.from_texts(
        chunks, embeddings, persist_directory="./chroma"
    )

    query = user_query
    relevant_chunks = get_similar(query, db)

    input = user_query
    ask_ques(input, messages, db)

if __name__ == "__main__":
    get_stuff(text, image_text, user_query, messages)