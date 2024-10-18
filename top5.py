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

client = chromadb.Client()
collection_name = "pdf_chunks"
collection = client.get_or_create_collection(collection_name)

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

def ask_for_questions():
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
    print(f"Generated Questions:\n{response}")

ask_for_questions()
