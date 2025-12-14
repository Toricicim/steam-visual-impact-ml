import requests
from bs4 import BeautifulSoup
import re

def analyze_steam_game(app_id):
    url = f"https://store.steampowered.com/app/{app_id}/?l=english"

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.5'
    }

    cookies = {'birthtime': '568028401', 'lastagecheckage': '1-0-1988'}

    try:
        response = requests.get(url, headers=headers, cookies=cookies)
        soup = BeautifulSoup(response.content, 'html.parser')

        title_tag = soup.find('div', {'class': 'apphub_AppName'})
        title = title_tag.text.strip() if title_tag else "Bilinmiyor"

        review_count = 0

        summary_rows = soup.find_all(
            'div', {'class': 'user_reviews_summary_row'})

        candidates = []
        for row in summary_rows:
            text = row.get_text()
            matches = re.findall(r'\(([\d,]+)\)', text)
            for match in matches:
                num = int(match.replace(',', ''))
                candidates.append(num)

        if candidates:
            review_count = max(candidates)

        if review_count == 0:
            meta_review = soup.find('meta', {'itemprop': 'reviewCount'})
            if meta_review:
                review_count = int(meta_review['content'])

        desc_div = soup.find('div', {'id': 'game_area_description'})

        gif_count = 0
        image_count = 0
        bullet_points = 0
        word_count = 0

        if desc_div:
            imgs = desc_div.find_all('img')
            for img in imgs:
                src = img.get('src', '')
                if 'gif' in src:
                    gif_count += 1
                else:
                    image_count += 1

            bullet_points = len(desc_div.find_all('li'))

            text_content = desc_div.get_text(separator=' ')
            word_count = len(text_content.split())

        demo_exists = 1 if soup.find(
            'div', {'class': 'demo_area_button'}) else 0

        glance_tags = soup.find_all('a', {'class': 'app_tag'})
        tags = [tag.text.strip() for tag in glance_tags]

        return {
            'Game': title,
            'Review_Count': review_count,
            'Gif_Count': gif_count,
            'Image_Count': image_count,
            'Bullet_Points': bullet_points,
            'Word_Count': word_count,
            'Has_Demo': demo_exists,
            'Tags': tags[:5]
        }

    except Exception as e:
        print(f"Hata: {e}")
        return None


game_data = analyze_steam_game(367520)  # Hollow Knight

if game_data:
    for key, value in game_data.items():
        print(f"{key}: {value}")
else:
    print("Error!!")
