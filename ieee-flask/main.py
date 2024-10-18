from flask import Flask, json, request, jsonify, render_template
import requests
import os
from initial import get_stuff
from input_output import get_message
app = Flask(__name__)

def download_pdf(url, save_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)
        return True
    return False

def search_in_chroma(query):
    """
    Performs a similarity search for the given query using the Chroma database.
    """
    # Generate embedding for the query
    query_embedding = embedder.encode(query, convert_to_tensor=False)
    
    # Get the total number of elements in the collection
    total_elements = collection.count()
    
    # Adjust n_results based on the total number of elements
    n_results = min(5, total_elements)
    
    # Perform similarity search in Chroma
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    
    # Return the top results along with their metadata
    return results["documents"], results["metadatas"]

@app.route('/initial', methods=['POST'])
def initial():
    data = request.json
    pdfUrl = data.get('pdfUrl')
    
    if not pdfUrl:
       
        return jsonify({'error': 'PDF URL is required'}), 400
    pdf_path = 'downloaded_resume.pdf'
    print("req here")
    if not download_pdf(pdfUrl, pdf_path):
        return jsonify({'error': 'Failed to download PDF'}), 500
   
    text, captions = get_stuff(pdf_path)
 
    os.remove(pdf_path)
    
    query = "Generate top 5 questions on the given context"
    message = [{"role":"system","content":"you are a helpful chatbot"}]
    res = get_message(text,captions,query,message)
    ques = res.split("\n")
    output = {'text': text, 'images': captions,'ques':ques}
    return jsonify(output)

@app.route('/message', methods=['POST'])
def message():
    data = request.json
    text = data.get('text')
    image_text = data.get('image_text')
    if not image_text:
        image_text = []
    user_query = data.get('userQuery')
    tempMessages = data.get('messages')
    print("messages: ",tempMessages)
    messages = list(map(lambda msg: {'role': msg['role'], 'content': msg['content']}, tempMessages))
    if not text:
        return jsonify({'error': 'text is required'}), 400
    
    if not user_query:
        return jsonify({'error': 'user_query is required'}), 400
    if not messages:
        return jsonify({'error': 'messages is required'}), 400
    print("messages: ",messages)
    message,metadata = get_message(text,image_text,user_query,messages)
    print("messages: ",messages)
    print("output: ",message)
    output = {'message':message,'metadata':metadata}
    return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True)