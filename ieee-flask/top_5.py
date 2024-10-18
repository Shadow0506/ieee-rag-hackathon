from langchain.prompts import ChatPromptTemplate
from groq import Groq
import chromadb
from sentence_transformers import SentenceTransformer

PROMPT_TEMPLATE = """
Answer the user query using only the relevant information provided to you in this prompt.

Relevant context from documents:
{relevant}

user query: {query}
"""

embedder = SentenceTransformer('all-MiniLM-L6-v2')

client = chromadb.Client()
collection_name = "pdf_chunks"
collection = client.get_or_create_collection(collection_name)

def get_message(text, image_text, user_query, messages):
    # Step 1: Add chunks of the PDF to Chroma
    add_text_chunks_to_chroma(text, image_text)

    # Step 2: Search for the question in the Chroma database
    documents, metadatas = search_in_chroma(user_query)

    return ask_ques(documents, user_query, messages),metadatas

    # Step 3: Print out the results with page numbers
    # for doc, meta in zip(documents[0], metadatas[0]):
    #     print(f"Found in page {meta['page_num']}: {doc}")

def chunk_text(text, chunk_size=500):
    """
    Splits the text into smaller chunks of the specified size.
    """
    words = text.split()
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def add_text_chunks_to_chroma(text, image_text):
    """
    Extracts text from a PDF, breaks it into chunks, generates embeddings,
    and stores them in the Chroma database.
    """
    pdf_text = text
    
    chunks = []
    metadata = []
    ids = []
    
    for i, (text, page_num) in enumerate(pdf_text):
        # Chunk the text into smaller pieces
        chunked_text = chunk_text(text)
        chunks.extend(chunked_text)
        
        # Store metadata with page number for each chunk
        metadata.extend([{"page_num": page_num}] * len(chunked_text))
        
        # Generate unique IDs for each chunk
        ids.extend([f"chunk_{i}_{j}" for j in range(len(chunked_text))])
    
    chunks = chunks + image_text

    # Generate embeddings for all chunks
    embeddings = embedder.encode(chunks, convert_to_tensor=False)
    
    # Add the chunks, metadata, embeddings, and IDs to the Chroma collection
    collection.add(
        ids=ids,
        documents=chunks,
        metadatas=metadata,
        embeddings=embeddings
    )

def search_in_chroma(query):
    """
    Performs a similarity search for the given query using the Chroma database.
    """
    # Generate embedding for the query
    query_embedding = embedder.encode(query, convert_to_tensor=False)
    
    # Perform similarity search in Chroma
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )
    
    # Return the top 5 results along with their metadata
    # return results["documents"], results["metadatas"]
    return results["documents"], results["metadatas"]

def ask_ques(results, user_query, messages):
    template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = template.format(
        relevant=results,
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

    return chat_completion.choices[0].message.content

def search_in_chroma(query):
    """
    Performs a similarity search for the given query using the Chroma database.
    """

    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    # Generate embedding for the query
    query_embedding = embedder.encode(query, convert_to_tensor=False)
    
    # Perform similarity search in Chroma
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )
    
    # Return the top 5 results along with their metadata
    return results["documents"], results["metadatas"]

def ask_for_questions(text):
    collection_name = "pdf_chunks"
    collection = client.get_or_create_collection(name=collection_name)
    messages = [{"role": "system", "content": "You are a helpful chatbot"}]
    template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    
    # Retrieve top 5 relevant chunks
    relevant_chunks, _ = search_in_chroma("Generate questions from this document")

    # Format the prompt with the relevant information
    prompt = template.format(
        relevant=relevant_chunks,
        query="Generate questions from this document"
    )

    print(f"promt: {prompt}")

    # Call the LLM API to generate questions
    groq_client = Groq(
        api_key="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",
    )

    messages.append({"role": "user", "content": prompt})

    chat_completion = groq_client.chat.completions.create(
        messages=messages,
        model="llama-3.1-8b-instant",
    )

    # Get and print the generated questions
    response = chat_completion.choices[0].message.content
    output = response.split("\n")
    return output



