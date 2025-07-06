import streamlit as st
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

import json5
import tqdm
import os
import re

# Page setup


st.set_page_config(page_title="Kaggle Feature Recommender", layout="wide")
st.title("🔮 Kaggle Feature Recommendation System")

st.markdown("""
    <style>
        .stApp {
            background-image: url("https://images.squarespace-cdn.com/content/v1/67e177fe3a167e189d4c96fc/6571b129-7948-4e9d-ab0e-101c96c39917/Gazza.png");
            background-size: 25%;
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-position: top right;
        }
        
        p {
            font-size: 18px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
This tool provides feature recommendations for Kaggle competitions based on competition overview and data description.
""")



# Initialize models (cached for performance)
@st.cache_resource
def load_models():
    try:
        # Load Sentence Transformer and FAISS index
        model = SentenceTransformer("all-MiniLM-L6-v2")
        index = faiss.read_index("models/rag_index/feature_index.faiss")
        
        # Load data
        with open("data/json_dataset/data_postprocessed_clean.json", "r", encoding="utf-8") as f:
            records = json.load(f)
        
        # Configure Gemini
        genai.configure(api_key=os.environ['GEMINI_KEY'])  # Use st.secrets for production!
        llm = genai.GenerativeModel("gemini-2.0-flash")
    
    except Exception as e:
        st.error(f"ERROR: {e}")
    return model, index, records, llm

model, index, records, llm = load_models()

# RAG system functions
def retrieve_relevant_docs(query: str, k: int = 5):
    """Retrieve most relevant documents using vector similarity"""
    query_embedding = model.encode(query)
    if query_embedding.ndim == 1:
        query_embedding = np.expand_dims(query_embedding, axis=0)
    distances, indices = index.search(query_embedding, k)
    return [records[i]["text"] for i in indices[0]], [records[i]["competition_slug"] for i in indices[0]]

def build_inference_prompt(new_comp_overview: str, new_comp_data_desc: str, contexts: list[str]) -> str:
    """Construct the LLM prompt with context"""
    context_block = "\n---\n".join(contexts)
    return f"""
    You are a Kaggle competition expert. Based on these previous competition discussions:
    {context_block}
    
    New competition overview:
    {new_comp_overview}
    
    Data description:
    {new_comp_data_desc}
    
    Generate a JSON array of feature recommendations, each with:
    - "feature_name": short name
    - "description": rationale
    
    Return ONLY valid JSON, no other text.
    """

def generate_answer(prompt: str):
    """Get response from Gemini LLM"""
    try:
        response = llm.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Generation error: {str(e)}")
        return None

def ask_rag_for_new_competition(new_comp_overview: str, new_comp_data_desc: str):
    """Main RAG pipeline function"""
    with st.spinner("🔍 Searching similar competitions..."):
        similar_docs, competition_idx = retrieve_relevant_docs(new_comp_overview)
    
    with st.spinner("🧠 Generating recommendations..."):
        prompt = build_inference_prompt(new_comp_overview, new_comp_data_desc, similar_docs)
        return generate_answer(prompt), competition_idx
    
    
    
def parse_json(raw_data):
    
    # results = []
    c = 0
    # for i, article in tqdm(enumerate(raw_data), desc="Processing articles", total=len(raw_data)):
    try:
        
        json_str = re.sub(r"^```json\s*|\s*```$", "", raw_data.strip())
        structured_info = json5.loads(json_str)
        
        
        structured_info
        
    except Exception as e:
        print(f"Error: {e}")
        c+=1
        # all_articles[i]['llm_report'] = analysis
        # continue
        
    return structured_info

# User Interface
with st.form("input_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        new_comp_overview = st.text_area(
            "**Competition Overview**",
            height=200,
            placeholder="Example: 'Predict flood probability based on environmental factors...'"
        )
    
    with col2:
        new_comp_data_desc = st.text_area(
            "**Data Description**",
            height=200,
            placeholder="Example: 'Dataset contains 50 features including rainfall, elevation...'"
        )
    
    submitted = st.form_submit_button("🚀 Generate Features")

# Results processing
if submitted:
    if not new_comp_overview or not new_comp_data_desc:
        st.warning("Please fill in both fields")
    else:
        result, similar_docs = ask_rag_for_new_competition(new_comp_overview, new_comp_data_desc)
        
        print(similar_docs)
        
        if result:
            try:
                features = parse_json(result)
                # features = json.loads(result)
                st.success("✅ Features generated!")
                
                st.subheader("Similar Competitions:")
                
                for i, comp in enumerate(similar_docs, 1):
                    with st.container():
                        comp_url = f"https://www.kaggle.com/competitions/{comp}"
                        st.markdown(f"""🔍 [**{comp}**]({comp_url})""", unsafe_allow_html=True)
                
                st.subheader("Recommended Features:")
                
                # print(features)
                
                for i, feature in enumerate(features, 1):
                    with st.expander(f"🌟 {feature.get('feature_name', 'Unnamed')}"):
                        st.write(feature.get('description', 'No description'))
                
                st.divider()
                # st.subheader("Model Response (JSON):")
                # st.code(result, language='json')
            except json.JSONDecodeError:
                st.error("Response format error. Received:")
                st.text(result)
        else:
            st.error("Failed to generate recommendations")

# Footer
st.sidebar.markdown("""
### About the System
Powered by:
- Sentence Transformer for search
- Gemini Pro for generation
- FAISS for vector search
""")