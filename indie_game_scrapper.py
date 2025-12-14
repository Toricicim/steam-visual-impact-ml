import csv
import time
import random
import requests
from bs4 import BeautifulSoup
import re
import os

INPUT_FILE = "steam_indie_ids.csv"
OUTPUT_FILE = "steam_games_dataset_clean.csv" 

# Filter Setting:
MIN_REVIEW_THRESHOLD = 20 

def get_api_data(app_id):
    try:
        price = 0.0
        api_reviews = 0
        api_ratio = 0.0
        has_demo_api = 0

        # 1. Price and Demo
        try:
            url = f"https://store.steampowered.com/api/appdetails?appids={app_id}&cc=us&filters=price_overview,demos"
            resp = requests.get(url, timeout=5)
            data = resp.json().get(str(app_id), {}).get('data', {})
            
            if data.get('price_overview'):
                price = data['price_overview']['final'] / 100.0
            if 'demos' in data:
                has_demo_api = 1
        except:
            pass

        # 2. (API)
        try:
            url = f"https://store.steampowered.com/appreviews/{app_id}?json=1&language=all&purchase_type=all"
            resp = requests.get(url, timeout=5)
            summary = resp.json().get('query_summary', {})
            
            api_reviews = summary.get('total_reviews', 0)
            total_pos = summary.get('total_positive', 0)
            if api_reviews > 0:
                api_ratio = total_pos / api_reviews
        except:
            pass

        return price, has_demo_api, api_reviews, api_ratio

    except:
        return 0.0, 0, 0, 0.0

def get_html_data(app_id):
    url = f"https://store.steampowered.com/app/{app_id}/?l=english"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    cookies = {'birthtime': '568028401', 'lastagecheckage': '1-0-1988'}
    
    try:
        response = requests.get(url, headers=headers, cookies=cookies, timeout=10)
        if response.status_code != 200: return None
        soup = BeautifulSoup(response.content, 'html.parser')
        
        title = soup.find('div', {'class': 'apphub_AppName'})
        title = title.text.strip() if title else "Unknown"

        html_reviews = 0
        html_ratio = 0.0
        
        summary_row = soup.find('div', {'class': 'user_reviews_summary_row'})
        if summary_row:
            text = summary_row.get_text()
            
            rev_match = re.search(r'([\d,]+)\s+user reviews', text)
            if rev_match:
                html_reviews = int(rev_match.group(1).replace(',', ''))
            
            ratio_match = re.search(r'(\d+)%', text)
            if ratio_match:
                html_ratio = int(ratio_match.group(1)) / 100.0
        
        if html_reviews == 0:
            meta = soup.find('meta', {'itemprop': 'reviewCount'})
            if meta: html_reviews = int(meta['content'])

        desc_div = soup.find('div', {'id': 'game_area_description'})
        gif_count = 0
        image_count = 0 
        bullet_points = 0
        word_count = 0
        
        if desc_div:
            imgs = desc_div.find_all('img')
            for img in imgs:
                src = (img.get('src') or img.get('data-src') or "").lower()
                if 'gif' in src:
                    gif_count += 1
                elif any(x in src for x in ['.jpg', '.jpeg', '.png', '.avif', '.webp']):
                    image_count += 1
            
            videos = desc_div.find_all('video')
            gif_count += len(videos) 
            
            bullet_points = len(desc_div.find_all('li'))
            word_count = len(desc_div.get_text(separator=' ').split())

        has_demo_html = 1 if (soup.find('div', {'class': 'demo_area_button'}) or soup.find('div', {'id': 'demo_block'})) else 0
        
        tags = "|".join([t.text.strip() for t in soup.find_all('a', {'class': 'app_tag'})][:5])
        
        return {
            'Title': title,
            'Html_Review': html_reviews,
            'Html_Ratio': html_ratio,  
            'Gif_Count': gif_count,
            'Image_Count': image_count,
            'Bullet_Points': bullet_points,
            'Word_Count': word_count,
            'Has_Demo_Html': has_demo_html,
            'Tags': tags
        }

    except Exception as e:
        return None

ids_to_scan = []
try:
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        r = csv.reader(f); next(r)
        for row in r: ids_to_scan.append(row[0])
except:
    print("ID dosyası yok.")
    exit()

headers = ['AppID', 'Title', 'Price', 'Review_Count', 'Positive_Ratio', 
           'Gif_Count', 'Image_Count', 'Bullet_Points', 'Word_Count', 'Has_Demo', 'Tags']

if not os.path.isfile(OUTPUT_FILE):
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()

print(f"--- TARAMA BAŞLIYOR (Filtre: <{MIN_REVIEW_THRESHOLD} inceleme 0 sayılacak) ---")

for i, app_id in enumerate(ids_to_scan):
    h_data = get_html_data(app_id)
    
    if h_data:
        price, api_demo, api_rev, api_ratio = get_api_data(app_id)
        
        final_review = max(api_rev, h_data['Html_Review'])
        
        final_ratio = api_ratio
        if final_ratio == 0.0 and h_data['Html_Ratio'] > 0:
            final_ratio = h_data['Html_Ratio']
            
        final_demo = max(api_demo, h_data['Has_Demo_Html'])


        # Filtering:
        if final_review < MIN_REVIEW_THRESHOLD:
            final_review = 0
            final_ratio = 0.0
            
        final_data = {
            'AppID': app_id,
            'Title': h_data['Title'],
            'Price': price,
            'Review_Count': final_review,
            'Positive_Ratio': round(final_ratio, 2),
            'Gif_Count': h_data['Gif_Count'],
            'Image_Count': h_data['Image_Count'],
            'Bullet_Points': h_data['Bullet_Points'],
            'Word_Count': h_data['Word_Count'],
            'Has_Demo': final_demo,
            'Tags': h_data['Tags']
        }
        
        with open(OUTPUT_FILE, 'a', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=headers)
            w.writerow(final_data)
            
        status = "OK" if final_review > 0 else "LOW DATA"
        print(f"[{i+1}] {final_data['Title'][:15]:<15} | Rev:{final_review:<5} | Ratio:{final_data['Positive_Ratio']:<4} | Gif+Vid:{final_data['Gif_Count']} -> {status}")
    else:
        print(f"[{i+1}] {app_id} ATLANDI")
        
    time.sleep(random.uniform(1.0, 1.5))

print("Done.")