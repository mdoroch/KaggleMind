from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from tqdm import tqdm
import time
import json
import argparse
import re
import os
from concurrent.futures import ThreadPoolExecutor


options = Options()
options.headless = True


driver = webdriver.Chrome(options=options)
wait = WebDriverWait(driver, 10)

def get_discussionsion(competition_slug):
    
    '''
    Get discussion links from a Kaggle competition leaderboard page.
    '''

    url = f"https://www.kaggle.com/competitions/{competition_slug}/leaderboard"

    driver.get(url)

    time.sleep(3)

    soup = BeautifulSoup(driver.page_source, "html.parser")
    
    discussion_links = soup.find_all("a", href=lambda x: x and f"/competitions/{competition_slug}/discussion/" in x)

    hrefs = [a["href"] for a in discussion_links]

    # driver.quit()
    
    return hrefs

def parse_discussion(href):
    '''
    Parse discussion links from the BeautifulSoup object.
    '''

    url = f"https://www.kaggle.com" + href

    driver.get(url)

    time.sleep(3)

    soup = BeautifulSoup(driver.page_source, "html.parser")

    # driver.quit()

    text = soup.get_text(separator=" ", strip=True)
    
    return text

def parse_competition_overview(competition_slug):
    '''
    Parse competition overview and evaluation.
    '''
    
    if '/' in competition_slug:
        
        competition_slug = re.search(r'/competitions/([^/?#]+)', competition_slug).group(1)
        
    url = f"https://www.kaggle.com/competitions/{competition_slug}/data"

    driver.get(url)

    time.sleep(3)

    soup = BeautifulSoup(driver.page_source, "html.parser")

    # driver.quit()
    
    return ' '.join([s.get_text(separator=" ", strip=True) for s in soup.find_all("p")])

def parse_competition_data_desc(competition_slug):

    if '/' in competition_slug:
        
        competition_slug = re.search(r'/competitions/([^/?#]+)', competition_slug).group(1)
    
    url = f"https://www.kaggle.com/competitions/{competition_slug}/overview"

    driver.get(url)

    time.sleep(3)

    soup = BeautifulSoup(driver.page_source, "html.parser")

    feature_description = soup.get_text(separator=" ", strip=True)
    
    return feature_description

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Scrape Kaggle discussion links from competition leaderboard pages.")

    parser.add_argument(
        "--output_file", 
        type=str, 
        default="tabular_classic_competitions.json",
        help="Name of the output JSON file"
    )

    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="data/raw_dataset", 
        help="Directory where the output file will be saved"
    )
    
    parser.add_argument(
        "--comp_list", 
        type=bool, 
        default=True, 
        help="use generated competition list or not (default: True)"
    )


    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    competitions = []
    
    if args.comp_list:
        
        with open('tabular_competitions.txt', "r", encoding="utf-8") as file:
            
            for line in file:
  
                competitions.append(line.strip())
        
    else:
        
        competitions_s3 = [f"playground-series-s3e{episode}" for episode in range(1, 27)]
        competitions_s4 = [f"playground-series-s4e{episode}" for episode in range(1, 13)]
        competitions_s5 = [f"playground-series-s5e{episode}" for episode in range(1, 13)]
        
        competitions = competitions_s3 + competitions_s4 + competitions_s5
    
    solutions = []
    

    for _, competition_slug in tqdm(enumerate(competitions), total=len(competitions)):
        
        
        output_path = os.path.join(args.output_dir, args.output_file)
        
        try:
            hrefs = get_discussionsion(competition_slug)
            
            overview = parse_competition_overview(competition_slug)
            
            data_desc = parse_competition_data_desc(competition_slug)

            # with ThreadPoolExecutor(max_workers=8) as executor:
            #     discussions = list(executor.map(parse_discussion, hrefs))
                
            discussions = [parse_discussion(href) for href in hrefs] #list(executor.map(parse_discussion, hrefs))
                
                

            solutions.append({
                'competition_slug': competition_slug,
                'discussion_links': hrefs,
                'discussion_texts': discussions, 
                'competition_overview': overview,
                'data_description': data_desc
            })
            
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(solutions, f, ensure_ascii=False, indent=2)


        except Exception as e:
            print(f"Error processing {competition_slug}: {e}")
            continue
        
    driver.quit()
        
    
