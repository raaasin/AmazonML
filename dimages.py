import os
import pandas as pd
import aiohttp
import asyncio
from tqdm.asyncio import tqdm
import urllib.parse

async def download_image(session, image_url, file_path, progress_bar):
    if os.path.exists(file_path):
        progress_bar.update(1)
        return True
    try:
        async with session.get(image_url) as response:
            if response.status == 200:
                img_data = await response.read()
                with open(file_path, 'wb') as file:
                    file.write(img_data)
                progress_bar.update(1)
                return True
            else:
                print(f"Failed to download image: {image_url} with status code {response.status}")
                return False
    except Exception as e:
        print(f"Error downloading image {image_url}: {e}")
        return False

async def download_images(image_urls, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    async with aiohttp.ClientSession() as session:
        progress_bar = tqdm(total=len(image_urls), desc="Downloading images", unit="image")
        tasks = []
        for url in image_urls:
            path = urllib.parse.urlsplit(url).path
            filename = os.path.basename(path)  
            if not filename:  
                filename = 'default_image.jpg'
            file_path = os.path.join(output_folder, filename)
            tasks.append(download_image(session, url, file_path, progress_bar))    
        results = await asyncio.gather(*tasks)
        progress_bar.close()
    return results

def main():
    DATASET_FOLDER = 'dataset/'
    csv_file = os.path.join(DATASET_FOLDER, 'new_train.csv')
    output_folder = 'timages'
    test = pd.read_csv(csv_file)
    image_urls = test['image_link'].tolist()
    asyncio.run(download_images(image_urls, output_folder))
    print("Download complete.")

if __name__ == "__main__":
    main()
