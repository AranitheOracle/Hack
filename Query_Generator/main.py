import torch
import serpapi
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import streamlit as st
from yahoo_search import search
import spacy
import re
from collections import Counter

# Cache the model loading to avoid re-downloading every time
@st.cache_resource
def load_model():
    # Initialize models and processor
    blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    blip_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    blip_model.to(device)
    
    return blip_processor, blip_model, device
nlp = spacy.load("en_core_web_sm")
# Define helper functions
def extract_search_prompt(full_string):
    parts = full_string.split("Query:")
    if len(parts) > 1:
        search_prompt = parts[-1].strip()
        return search_prompt
    return None 
def identify_important_noun(sentence):
    doc = nlp(sentence)
    noun_freq = Counter(token.text.lower() for token in doc if token.pos_ == "NOUN")
    if not noun_freq:
        return None
    most_common_noun, _ = noun_freq.most_common(1)[0]
    return most_common_noun
def extract_keywords(image_description, prompt, image_main):
    image_doc = nlp(image_description.lower())
    prompt_doc = nlp(prompt.lower())
    image_main = nlp(image_main.lower())
    objects_in_image = [token.text for token in image_doc if token.pos_ == "NOUN"]
    attributes_in_image = [token.text for token in image_doc if token.pos_ == "ADJ"]
    objects_in_main = [token.text for token in image_main if token.pos_ == "NOUN"]
    attribute_in_main = [token.text for token in image_main if token.pos_ == "ADJ"]
    objects_in_prompt = [token.text for token in prompt_doc if token.pos_ == "NOUN"]
    attribute_in_prompt = [token.text for token in prompt_doc if token.pos_ == "ADJ"]
    print(objects_in_image,attributes_in_image,objects_in_prompt,attribute_in_prompt, objects_in_main, attribute_in_main)
    return (objects_in_image,attributes_in_image,objects_in_prompt,attribute_in_prompt, objects_in_main, attribute_in_main)
def is_proper_noun(word):
    doc = nlp(word)
    for token in doc:
        if token.pos_ == "PROPN":  
            return True
    return False    

def finalize(image_description,user_prompt, image_main):
    image_description = image_description.lower()
    user_prompt = re.sub(r'[^A-Za-z\s]', '', user_prompt)
    user_prompt= user_prompt.lower()

    o_i,a_i,o_p,a_p,o_m,a_m =extract_keywords(image_description, user_prompt, image_main)
    main_prompt=""
    main_desc =""
    main_attributes = ""
    for i in user_prompt.split():

        if(i in o_p) or (i in a_p):
            
            main_prompt+=i
            main_prompt += " "

    for i in image_description.split():

        if(i in o_i) or (i in a_i):
            
            main_desc+=i
            main_desc+= " "
        if((i in o_i) or (i in a_i)) and (i not in o_m ) and (i not in a_m):
            main_attributes+=i
            
            main_attributes+=" "
    return main_prompt, main_desc
def generate_prompt(image, text, blip_processor, blip_model, device):
    inputs_ = blip_processor(image, text="Question: What is the main object here? Answer:", return_tensors="pt").to(device, torch.float16)
    generated_ids_ = blip_model.generate(**inputs_)
    generated_main_details = blip_processor.batch_decode(generated_ids_, skip_special_tokens=True)[0].strip()
     
    inputs = blip_processor(image, return_tensors="pt").to(device, torch.float16)
    generated_ids = blip_model.generate(**inputs)
    generated_description = blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    prompt,desc = finalize(generated_description,text,generated_main_details)
    final_response = finalize_prompt(generated_main_details, desc, prompt)
    return final_response

def finalize_prompt(main_details, description, prompt):
    final_response = prompt + " given the image shows " + description
    return final_response

def search_yahoo(query):
    result = search(query)
    return result.pages[0]

def search_web():
   

    params = {
    "engine": "yahoo",
    "query": "show me a pen of red colour like this car",
    "api_key": "625979961b68a67f3d06e40c23682c42c91b80fdf4020c419c5a2224488f146e"
    }

    results = serpapi.search(params)
   
    images_results = results["inline_images"]
    images_results = (images_results['items'])
    return images_results

# Main function for Streamlit UI
def main():
    st.title("Image to Web Search")

    # Load the model once using the cache
    blip_processor, blip_model, device = load_model()

    # Image uploader
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    # Text input for the query
    query_text = st.text_input("Enter the query description")

    if st.button("Generate Web Search"):
        if uploaded_image and query_text:
            # Load the image
            image = Image.open(uploaded_image)

            # Generate the search query
            query = generate_prompt(image, query_text, blip_processor, blip_model, device)

            # Show the generated search query
            st.write(f"Generated Search Query: {query}")

            # Search the web and display the results
            if query:
                search_results = search_web(query)

                # Display the search results
                if search_results:
                    st.image([result['thumbnail'] for result in search_results], width=150)
                else:
                    st.write("No results found.")
        else:
            st.write("Please upload an image and enter a query.")

# Run the app
if _name_ == "_main_":
    main()