import json
import faiss
import numpy as np
import os
import streamlit as st
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
# from helper_func.prompt_builder import build_inference_prompt


import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
libs_dir = os.path.join(current_dir, "../")
sys.path.append(os.path.abspath(libs_dir))

from helper_func.prompt_builder import build_postprocess_prompt, build_inference_prompt

# Setup SentenceTransformer + FAISS


# model = SentenceTransformer("all-MiniLM-L6-v2")
# index = faiss.read_index("models/rag_index/feature_index.faiss")

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
        
        genai.configure(api_key=st.secrets["api_keys"]["GEMINI_KEY"])

        model_name = "gemini-2.0-flash"

        llm = genai.GenerativeModel(model_name)
    
    except Exception as e:
        st.error(f"ERROR: {e}")
        
    return model, index, records, llm

model, index, records, llm = load_models()

# Load mapped texts
# with open("records.json", "r", encoding="utf-8") as f:
#     records = json.load(f)
    
    
# with open("data/json_dataset/data_postprocessed_clean.json", "r", encoding="utf-8") as f:
#     records = json.load(f)

# Configure Gemini
# genai.configure(api_key=os.environ['GEMINI_KEY'])

# genai.configure(api_key=st.secrets["api_keys"]["GEMINI_KEY"])

# model_name = "gemini-2.0-flash"

# llm = genai.GenerativeModel(model_name)

def retrieve_relevant_docs(query: str, k: int = 5):
    """Retrieve most relevant documents using vector similarity"""
    query_embedding = model.encode(query)
    if query_embedding.ndim == 1:
        query_embedding = np.expand_dims(query_embedding, axis=0)
    distances, indices = index.search(query_embedding, k)
    return [records[i]["text"] for i in indices[0]], [records[i]["competition_slug"] for i in indices[0]]

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
        similar_docs, competition_idx = retrieve_relevant_docs(new_comp_overview, k = 3)
    
    with st.spinner("🧠 Generating recommendations..."):
        prompt = build_inference_prompt(new_comp_overview, new_comp_data_desc, similar_docs)
        return generate_answer(prompt), competition_idx


# # Example usage
# if __name__ == "__main__":
#     # new_comp_overview = """Welcome to the 2024 Kaggle Playground Series! We plan to continue in the spirit of previous playgrounds, providing interesting and approachable datasets for our community to practice their machine learning skills, and anticipate a competition each month.
#     #                         Your Goal: The goal of this competition is to predict the probability of a region flooding based on various factors"""
                            
                            
#     new_comp_overview = """Overview
#             In this competition, hosted by Jane Street, you'll build a model using real-world data derived from production systems, which offers a glimpse into the daily challenges of successful trading. This challenge highlights the difficulties in modeling financial markets, including fat-tailed distributions, non-stationary time series, and sudden shifts in market behavior.
#             Description
#             When approaching modeling problems in modern financial markets, there are many reasons to believe that the problems you are trying to solve are impossible. Even if you put aside the beliefs that the prices of financial instruments rationally reflect all available information, you’ll have to grapple with time series and distributions that have properties you don’t encounter in other sorts of modeling problems. Distributions can be famously fat-tailed, time series can be non-stationary, and data can generally fail to satisfy a lot of the underlying assumptions on which very successful statistical approaches rely. Layer on all of this the fact that the financial markets are ultimately a human endeavor involving a large number of individuals and institutions that are constantly changing with advances in technology and shifts in society, and responding to economic and geopolitical issues as they arise - and you can start to get a sense of the difficulties involved!

#             In this challenge, we ask you to build a model using real-world data derived from some of our production systems. This data gives a very close picture of some of the things we have to do every day to be successful at trading in modern financial markets. We’ve assembled a collection of features and responders related to markets where we run automated trading strategies and are concerned about having good underlying models. To balance crafting a challenging, relevant problem that ties into our business while respecting the proprietary and highly competitive nature of our trading, you will notice that we have anonymized and lightly obfuscated some of the features and responders we present in the data. These modifications don’t change the essence of the problem at hand, but they do allow us to give you a difficult task that meaningfully illustrates the work we do at Jane Street.

#             Jane Street has spent decades relentlessly innovating on all aspects of our trading, and building machine learning models to aid our decision-making. These models help us actively trade thousands of financial products each day across 200+ trading venues around the world. While this challenge only presents a tiny fraction of the quantitative problems Jane Streeters work on daily, we are very interested in seeing how the Kaggle community will approach this challenge, and in engaging with you about your solutions to the problem!

#             Evaluation
#             Submissions are evaluated on a scoring function defined as the sample weighted zero-mean R-squared score (
#             ) of responder_6. The formula is give by:


#             where 
#             and 
#             are the ground-truth and predicted value vectors of responder_6, respectively; 
#             is the sample weight vector."""
#     # new_comp_data_desc = """Dataset Description
#     #                         The dataset for this competition (both train and test) was generated from a deep learning model trained on the Flood Prediction Factors dataset. Feature distributions are close to, but not exactly the same, as the original. Feel free to use the original dataset as part of this competition, both to explore differences as well as to see whether incorporating the original in training improves model performance.

#     #                         Note: This dataset is particularly well suited for visualizations, clustering, and general EDA. Show off your skills!

#     #                         Files
#     #                         train.csv - the training dataset; FloodProbability is the target
#     #                         test.csv - the test dataset; your objective is to predict the FloodProbability for each row
#     #                         sample_submission.csv - a sample submission file in the correct format"""
    
#     new_comp_data_desc = """Dataset Description
#             The competition dataset comprises a set of timeseries with 79 features and 9 responders, anonymized but representing real market data. The goal of the competition is to forecast one of these responders, i.e., responder_6, for up to six months in the future.

#             You must submit to this competition using the provided Python evaluation API, which serves test set data one timestep by timestep. To use the API, follow the example in this notebook. (Note that this API is different from our legacy timeseries API used in past forecasting competitions.)

#             Competition Phases and Data Updates
#             In line with the forecasting task, the competition will proceed in two phases:

#             A model training phase with a test set of historical data. This test set has about 4.5 million rows.
#             A forecasting phase with a test set to be collected after submissions close. You should expect this test set to be about the same size as the test set in the first phase.
#             To help you author robust submissions, during the final weeks of the model training phase we will be extending the public test set to include data closer to the submission deadline. Predictions on this extended set will not be scored.

#             At the start of the forecasting phase, the unscored public test set will be extended up to the final day of the model training phase and the private set updated roughly every two weeks. Submissions will be rescored at the time of each update.

#             During the forecasting phase, the evaluation API will serve test data from the beginning of the public set to the end of the private set. You must make predictions at every timestep, but, in this phase, only predictions on the private set are scored. (You may predict 0.0 on the unscored segments, if you like.)

#             File and Field Information
#             train.parquet - The training set, contains historical data and returns. For convenience, the training set has been partitioned into ten parts.
#             date_id and time_id - Integer values that are ordinally sorted, providing a chronological structure to the data, although the actual time intervals between time_id values may vary.
#             symbol_id - Identifies a unique financial instrument.
#             weight - The weighting used for calculating the scoring function.
#             feature_{00...78} - Anonymized market data.
#             responder_{0...8} - Anonymized responders clipped between -5 and 5. The responder_6 field is what you are trying to predict.
#             test.parquet - A mock test set which represents the structure of the unseen test set. This example set demonstrates a single batch served by the evaluation API, that is, data from a single date_id, time_id pair. The test set contains columns including date_id, time_id, symbol_id, weight, is_scored, and feature_{00...78}. You will not be directly using the test set or sample submission in this competition, as the evaluation API will get/set the test set and predictions.
#             is_scored - Indicates whether this row is included in the evaluation metric calculation.
#             lags.parquet - Values of responder_{0...8} lagged by one date_id. The evaluation API serves the entirety of the lagged responders for a date_id on that date_id's first time_id. In other words, all of the previous date's responders will be served at the first time step of the succeeding date.
#             sample_submission.csv - This file illustrates the format of the predictions your model should make.
#             features.csv - metadata pertaining to the anonymized features
#             responders.csv - metadata pertaining to the anonymized responders
#             Each row in the {train/test}.parquet dataset corresponds to a unique combination of a symbol (identified by symbol_id) and a timestamp (represented by date_id and time_id). You will be provided with multiple responders, with responder_6 being the only responder used for scoring. The date_id column is an integer which represents the day of the event, while time_id represents a time ordering. It's important to note that the real time differences between each time_id are not guaranteed to be consistent.

#             The symbol_id column contains encrypted identifiers. Each symbol_id is not guaranteed to appear in all time_id and date_id combinations. Additionally, new symbol_id values may appear in future test sets."""
                            
#     features_recommendation = ask_rag_for_new_competition(new_comp_overview, 
#                                                           new_comp_data_desc)
#     print(features_recommendation)