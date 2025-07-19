from google import genai
import json
import argparse

import re

import json5
from tqdm import tqdm

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
libs_dir = os.path.join(current_dir, "../../")
sys.path.append(os.path.abspath(libs_dir))

from helper_func.prompt_builder import build_postprocess_prompt


def generate_summary(client,
                     competition_slug: str, 
                     discussion_texts: str,
                     competition_overview: str,
                     data_description: str,
                     model_name = "gemini-2.0-flash") -> str:
    
    """
    postprocess the data using LLM
    """
    
    prompt = build_postprocess_prompt(competition_slug,
                                 competition_overview,
                                 data_description,
                                 discussion_texts)

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
    )

    return response.text


# with open('../json_dataset/data_postprocessed.json', "r", encoding="utf-8") as f:
#     raw_data = json.load(f)


def parse_json(raw_data):
    
    results = []
    c = 0
    for i, article in tqdm(enumerate(raw_data), desc="Processing articles", total=len(raw_data)):
        try:
            # full_text = article['title'] + '. ' + article['text']  # или article['full_text'] если уже есть
            # analysis = analyze_news_with_llm(client, model, full_text)
            
            json_str = re.sub(r"^```json\s*|\s*```$", "", article.strip())
            structured_info = json5.loads(json_str)
            
            structured_info['text'] = f'''Overview: {structured_info['overview_summary']},
                Data description: {structured_info['data_description_clean']},
                Feature insights: {structured_info['feature_insights']},
                Modeling strategies: {structured_info['modeling_strategies']},
                '''
            results.append(structured_info)
            
        except Exception as e:
            print(f"Error: {e}")
            c+=1
            # all_articles[i]['llm_report'] = analysis
            continue
        
    return results
    
    
if __name__ == "__main__":
    
    client = genai.Client(api_key=os.environ['GEMINI_KEY'])
    
    parser = argparse.ArgumentParser(description="Scrape Kaggle discussion links from competition leaderboard pages.")

    parser.add_argument(
        "--output_file", 
        type=str, 
        default="data_postprocessed.json",
        help="Name of the output JSON file"
    )
    
    parser.add_argument(
        "--output_file_clean", 
        type=str, 
        default="data_postprocessed_clean_test.json",
        help="Name of the output JSON file"
    )

    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="data/json_dataset", 
        help="Directory where the output file will be saved"
    )
    
    parser.add_argument(
        "--data_path", 
        type=str, 
        default="data/raw_dataset/tabular_classic_competitions.json", 
        help="Directory where the dataset file was saved"
    )

    args = parser.parse_args()
    
    with open(args.data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
        
    # print(raw_data)
        
    list_of_summaries = []
    
    for _, data in tqdm(enumerate(raw_data), total=len(raw_data)):
        
    # for _, competition_slug in tqdm(enumerate(competitions), total=len(competitions)):
        
        competition_slug = data['competition_slug']
        discussion_links = data['discussion_links']
        discussion_texts = ''.join(data['discussion_texts'])
        competition_overview = data['competition_overview']
        data_description = data['data_description']
        
        
        # Assuming the data structure is correct
        
        summary = generate_summary(client,
                                   competition_slug,
                                   discussion_texts,
                                   competition_overview,
                                   data_description)
        
        list_of_summaries.append(summary)
        
    output_path = os.path.join(args.output_dir, args.output_file)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(list_of_summaries, f, ensure_ascii=False, indent=2)
        
        
    cleaned_data = parse_json(list_of_summaries)
    
    output_path = os.path.join(args.output_dir, args.output_file_clean)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
            
    