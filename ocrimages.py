import os
from PIL import Image
import urllib.parse
import easyocr
import pandas as pd
import torch

count=0
print(torch.cuda.is_available())
reader = easyocr.Reader(['en'], gpu=True)

def get_image_filename(image_url):
    output_folder = 'images'
    path = urllib.parse.urlsplit(image_url).path
    filename = os.path.basename(path) 
    return os.path.join(output_folder, filename)

def open_image(image_url):
    file_path = get_image_filename(image_url)
    img = Image.open(file_path)
    return img

def extract_text_from_image(image):
    try:
        text = reader.readtext(image, detail=0, text_threshold=0.6) 
        return ' '.join(text)
    except Exception as e:
        return ""

def predictor(image_link, category_id, entity_name):
    global count
    count+=1
    try:
        image = open_image(image_link)
        extracted_text = extract_text_from_image(image)
        print([(count),(entity_name),(extracted_text)])
    except Exception as e:
        print(f"Error processing {image_link}: {e}")
        return ""

if __name__ == "__main__":
    DATASET_FOLDER = 'dataset/' 
    test = pd.read_csv(os.path.join(DATASET_FOLDER, 'test.csv')) 
    test['prediction'] = test.apply(
        lambda row: predictor(row['image_link'], row['group_id'], row['entity_name']), axis=1) 
    output_filename = 'test_out.csv'
    test[['index', 'prediction']].to_csv(output_filename, index=False)