import torch
import serpapi
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import streamlit as st
from langchain.llms import HuggingFaceHub

# Initialize models and processor
blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
blip_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
model_name = "Qwen/Qwen2.5-72B-Instruct"
access_token = "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"
llm = HuggingFaceHub(
    repo_id=model_name,
    huggingfacehub_api_token=access_token
)
device = "cuda" if torch.cuda.is_available() else "cpu"
blip_model.to(device)

# Define helper functions
def extract_search_prompt(full_string):
    parts = full_string.split("Query:")
    if len(parts) > 1:
        search_prompt = parts[-1].strip()
        return search_prompt
    return None 

def generate_prompt(image, text):
    inputs_ = blip_processor(image, text="Question: What is the main object here? Answer:", return_tensors="pt").to(device, torch.float16)
    generated_ids_ = blip_model.generate(**inputs_)
    generated_main_details = blip_processor.batch_decode(generated_ids_, skip_special_tokens=True)[0].strip()
     
    inputs = blip_processor(image, return_tensors="pt").to(device, torch.float16)
    generated_ids = blip_model.generate(**inputs)
    generated_description = blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    
    response = llm(f"Give me a search prompt for web search combining information. Make it short and focus on what object is being asked for in the query. Output: image description= {generated_description}, query description= {text}")
    
    final_response = extract_search_prompt(response)
    return final_response

def search_web(query):
    params = {
        "engine": "duckduckgo",
        "q": query,
        "api_key": "625979961b68a67f3d06e40c23682c42c91b80fdf4020c419c5a2224488f146e"
    }

    search = serpapi.search(params)
    
    inline_images = search["inline_images"]
    return inline_images

# Main function for Streamlit UI
def main():
    st.title("Image to Web Search")

    # Image uploader
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    # Text input for the query
    query_text = st.text_input("Enter the query description")

    if st.button("Generate Web Search"):
        if uploaded_image and query_text:
            # Load the image
            image = Image.open(uploaded_image)

            # Generate the search query
            query = generate_prompt(image, query_text)

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