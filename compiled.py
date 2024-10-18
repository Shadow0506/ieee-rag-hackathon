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

def generate_caption(image):
    # Load the BLIP model and processor
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    # Preprocess the image
    inputs = processor(image, return_tensors="pt")

    # Generate caption
    with torch.no_grad():
        outputs = model.generate(**inputs)

    # Decode the generated caption
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption

def extract_images_from_pdf(pdf_path):
    pdf_document = fitz.open(pdf_path)
    descriptions = {}

    for page_number in range(len(pdf_document)):
        page = pdf_document[page_number]
        image_list = page.get_images(full=True)

        for img_index, img in enumerate(image_list):
            xref = img[0]
            image = pdf_document.extract_image(xref)
            image_bytes = image["image"]

            # Open the image using PIL
            img = Image.open(io.BytesIO(image_bytes))

            # Generate caption using BLIP
            caption = generate_caption(img)  # Pass the image for captioning
            descriptions[f"Image {page_number + 1}.{img_index + 1}"] = caption

    return descriptions

# pdf_path = "C:\\Users\gantr\Downloads\Education_Can_t_Wait.pdf"  # Replace with your PDF file path

def extract_text_from_pdf(pdf_path):
    # Create a PdfReader object
    reader = PdfReader(pdf_path)

    # Initialize a string to hold all the extracted text
    extracted_text = ""

    # Loop through each page and extract text
    for page in reader.pages:
        extracted_text += page.extract_text()

    # Print the extracted text
    # with open('extracted_text.txt', 'w') as text_file:
        # text_file.write(extracted_text)

    return extracted_text

def ask_ques(user_query):
    template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = template.format(
        relevant=get_similar(query),
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
    messages.append({"role": "user", "content": "user_query"})
    messages.append({"role": "assistant", "content": chat_completion.choices[0].message.content})

    print(f"\n\nResponse: {chat_completion.choices[0].message.content}")

def get_stuff(pdf_path):
    
    text = extract_text_from_pdf(pdf_path)
    captions = extract_images_from_pdf(pdf_path)

    return text, captions

    # text_splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=500,
    #     chunk_overlap=200,
    #     length_function=len,
    #     add_start_index=True,
    # )

    # chunks = text_splitter.split_text(text)
    # print(f"Split document into {len(chunks)} chunks.")

    # embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"trust_remote_code":True}) 

    # db = Chroma.from_texts(
    #     chunks, embeddings, persist_directory="./chroma"
    # )

    # def get_similar(query):
    #     result = db.similarity_search_with_relevance_scores(query=query, k=10)
    #     return result

    # query = "What are the main points of the document?"
    # relevant_chunks = get_similar(query)
    # print(relevant_chunks)

    # PROMPT_TEMPLATE = """
    # Answer the user query using only the relevant information provided to you in this prompt.

    # Relevant context from documents:
    # {relevant}

    # user query: {query}
    # """

    # messages = [{"role": "system", "content": "You are a helpful chatbot"}]

    # input = "Tell me about bitcoin volatility"
    # ask_ques(input)