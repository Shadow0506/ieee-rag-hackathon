import fitz  # PyMuPDF
from pypdf import PdfReader
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import io

def generate_caption(image, max_new_tokens=50):
    # Load the BLIP model and processor
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    # Preprocess the image
    inputs = processor(image, return_tensors="pt")

    # Generate caption
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)

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
    """
    Extracts text from each page of the PDF and returns it as a list of tuples,
    where each tuple contains the text of a page and its corresponding page number.
    """
    text_per_page = []
    
    # Open the PDF file
    reader = PdfReader(pdf_path)
    
    # Loop through all pages and extract text
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text()  # Extract text from the page
        if text:  # Check if the page has any text
            text_per_page.append([text, str(page_num)])  # Append tuple (text, page number as string)
    
    return text_per_page

def get_stuff(pdf_path):
    
    text = extract_text_from_pdf(pdf_path)
    captions_dict = extract_images_from_pdf(pdf_path)
    captions =[]
    for key,val in captions_dict.items():
        captions.append(key+": "+val)
    print("type of ", type(captions))
    return text, captions
