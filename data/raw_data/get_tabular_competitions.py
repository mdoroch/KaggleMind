from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import time

def compet_parser(soup):
    '''
    Parse competition title from the overview page.
    '''
    links = soup.find_all('a', attrs={'aria-label': True})

    competition_links = []

    for link in links:
        href = link.get('href')
        aria_label = link.get('aria-label')
        
        if href.startswith('/competitions/'):
            competition_links.append(href.replace('/competitions/', ''))

    return competition_links

if __name__ == "__main__":
    
    print('ok')

    options = Options()
    options.headless = True
    driver = webdriver.Chrome(options=options)


    i = 1
    competition_links = []
    
    while True:

        file_path = "tabular_competitions.txt"
        
        url = f"https://www.kaggle.com/competitions?tagIds=14101&listOption=completed&hostSegmentIdFilter=1&page={i}"
        
        driver.get(url)

        time.sleep(5)

        soup = BeautifulSoup(driver.page_source, "html.parser")
        
        page_links = compet_parser(soup)
        
        if len(page_links) == 0:
            print(f"No more competitions found on page {i}.")
            
            
            with open(file_path, "w", encoding="utf-8") as file:
                for link in competition_links:
                    
                    file.write(link + "\n")
            break
        
        competition_links.extend(page_links)

        i+=1
        
        # with open("rendered_page.html", "w", encoding="utf-8") as f:
        #     f.write(soup.prettify())


            
    driver.quit()