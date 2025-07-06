from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
from tqdm import tqdm
import json
import argparse
import os
from concurrent.futures import ThreadPoolExecutor

def fetch_html(url):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, wait_until="networkidle")
        html = page.content()
        browser.close()
        return html

def get_discussion_links(competition_slug):
    '''
    Get discussion links from a Kaggle competition leaderboard page.
    '''
    url = f"https://www.kaggle.com/competitions/{competition_slug}/leaderboard"
    html = fetch_html(url)
    soup = BeautifulSoup(html, "html.parser")
    discussion_links = soup.find_all("a", href=lambda x: x and f"/competitions/{competition_slug}/discussion/" in x)
    hrefs = list(set(a["href"] for a in discussion_links if a.has_attr("href")))
    return hrefs

def parse_discussion(href):
    '''
    Parse discussion text from the full discussion page.
    '''
    url = f"https://www.kaggle.com{href}"
    html = fetch_html(url)
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator=" ", strip=True)

def parse_competition_overview(competition_slug):
    '''
    Parse competition title from the overview page.
    '''
    url = f"https://www.kaggle.com/competitions/{competition_slug}/overview"
    html = fetch_html(url)
    soup = BeautifulSoup(html, "html.parser")
    title_tag = soup.find("h1", class_="competition-title")
    return title_tag.get_text(strip=True) if title_tag else ""

def parse_competition_data_desc(competition_slug):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape Kaggle discussion links and content from leaderboard pages.")

    parser.add_argument(
        "--output_file",
        type=str,
        default="discussion_links_playground_pl.json",
        help="Name of the output JSON file"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="../raw_dataset",
        help="Directory where the output file will be saved"
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_file)

    competitions_s3 = [f"playground-series-s3e{episode}" for episode in range(1, 27)]
    competitions_s4 = []#[f"playground-series-s4e{episode}" for episode in range(1, 13)]
    competitions_s5 = []#[f"playground-series-s5e{episode}" for episode in range(1, 13)]

    competitions = competitions_s3 + competitions_s4 + competitions_s5
    solutions = []

    for _, competition_slug in tqdm(enumerate(competitions), total=len(competitions)):
        try:
            hrefs = get_discussion_links(competition_slug) #[:5]
            # with ThreadPoolExecutor(max_workers=8) as executor:
            discussions = [parse_discussion(href) for href in hrefs] #list(executor.map(parse_discussion, hrefs))

            solutions.append({
                'competition_slug': competition_slug,
                'discussion_links': hrefs,
                'discussion_texts': discussions
            })
            
            
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(solutions, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"Error processing {competition_slug}: {e}")
            continue

