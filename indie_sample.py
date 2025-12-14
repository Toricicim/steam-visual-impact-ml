import requests
from bs4 import BeautifulSoup
import time
import random
import csv

TARGET_COUNT = 1500
PAGES_TO_SCAN = 20  

URLS = {
    "Top_Rated": "https://store.steampowered.com/search/?tags=492&category1=998&sort_by=Reviews_DESC",
    "New_Releases": "https://store.steampowered.com/search/?tags=492&category1=998&sort_by=Released_DESC",
    "Relevance": "https://store.steampowered.com/search/?tags=492&category1=998" 
}

collected_ids = set()

def get_game_ids_randomly(base_url, num_pages_to_visit):
    count_before = len(collected_ids)
    
    headers = {'User-Agent': 'Mozilla/5.0 ...'} 
    cookies = {'birthtime': '568028401', 'lastagecheckage': '1-0-1988'}
    
    if "Reviews_DESC" in base_url:
        page_indices = range(0, num_pages_to_visit)
    else:
        # exp: Page 5, Page 42, Page 89...
        random_pages = random.sample(range(1, 150), num_pages_to_visit)
        page_indices = sorted(random_pages) 

    for page_num in page_indices:
        start_index = page_num * 50
        target_url = f"{base_url}&start={start_index}"
        
        try:
            response = requests.get(target_url, headers=headers, cookies=cookies)
            if response.status_code != 200: continue
                
            soup = BeautifulSoup(response.content, 'html.parser')
            rows = soup.find_all('a', {'class': 'search_result_row'})
            
            if not rows:
                continue 
                
            for row in rows:
                app_id = row.get('data-ds-appid')
                if app_id:
                    clean_id = app_id.split(',')[0]
                    collected_ids.add(clean_id)
            
            time.sleep(random.uniform(1.5, 3.0))
            
        except Exception as e:
            print(f"   Hata: {e}")
            
    print(f"   + {len(collected_ids) - count_before} yeni oyun eklendi.")

print(f"Hedef: {TARGET_COUNT} ID (Rastgele Sayfalama Aktif)")

for cat, url in URLS.items():
    print(f"\n[{cat}] Modu...")
    get_game_ids_randomly(url, PAGES_TO_SCAN)
    if len(collected_ids) >= TARGET_COUNT: break

with open("steam_indie_ids.csv", 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['AppID'])
    for app_id in collected_ids:
        writer.writerow([app_id])
print("Bitti.")