import os
import time
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

TARGET_DIR = "/Users/prabanandsc/C_Work/Sivaranjani_Work_1/Datasets/Census of India 2011"
os.makedirs(TARGET_DIR, exist_ok=True)

def download_file(url, target_path):
    if os.path.exists(target_path):
        print(f"Skipping already downloaded file: {target_path}")
        return
    print(f"Downloading: {url}")
    try:
        with requests.get(url, stream=True, verify=False, timeout=60) as r:
            r.raise_for_status()
            with open(target_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"Saved: {target_path}")
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        if os.path.exists(target_path):
            os.remove(target_path)

def main():
    page = 1
    page_size = 100
    base_url = "https://censusindia.gov.in/nada/index.php/api/catalog/search/?ps={}&page={}&sk=District%20Census%20Handbook%202011&include_resources=true"
    
    session = requests.Session()
    
    total_downloaded = 0
    while True:
        url = base_url.format(page_size, page)
        print(f"Fetching page {page}...")
        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = session.get(url, verify=False, timeout=30)
                data = resp.json()
                break
            except Exception as e:
                print(f"Error fetching page {page} (attempt {attempt+1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    print("Max retries reached. Stopping.")
                    break
                time.sleep(5)
        else:
            break
            
        rows = data.get("result", {}).get("rows", [])
        if not rows:
            print("No more results found.")
            break
            
        for row in rows:
            title = row.get("title", "Unknown_Title").replace("/", "_").replace("\\", "_")
            resources = row.get("resources", [])
            for res in resources:
                link = res.get("link")
                filename = res.get("title")
                if not filename:
                    filename = res.get("filename", f"{title}.file")
                if link:
                    # Sanitize filename
                    filename = filename.replace("/", "_").replace("\\", "_")
                    target_path = os.path.join(TARGET_DIR, filename)
                    download_file(link, target_path)
                    total_downloaded += 1
                    time.sleep(1) # Be polite to the server
                    
        page += 1

    print(f"Finished. Total resources processed: {total_downloaded}")

if __name__ == "__main__":
    main()
