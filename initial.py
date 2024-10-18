import fitz  # PyMuPDF
from pypdf import PdfReader
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import io

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

def get_stuff(pdf_path):
    
    text = extract_text_from_pdf(pdf_path)
    captions = extract_images_from_pdf(pdf_path)

    return text, captions
