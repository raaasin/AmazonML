import os
import requests
from PIL import Image
from io import BytesIO
import easyocr
import re
import pandas as pd
import torch
from multiprocessing import Pool, cpu_count

print(torch.cuda.is_available())
reader = easyocr.Reader(['en'], gpu=True)

def download_image(image_url):
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    return img

def extract_text_from_image(image):
    text = reader.readtext(image, detail=0, text_threshold=0.6) 
    return ' '.join(text)

def process_text(entity_name, extracted_text):
    entity_unit_map = {
        'width': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
        'depth': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
        'height': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
        'item_weight': {'gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton'},
        'maximum_weight_recommendation': {'gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton'},
        'voltage': {'kilovolt', 'millivolt', 'volt'},
        'wattage': {'kilowatt', 'watt'},
        'item_volume': {'centilitre', 'cubic foot', 'cubic inch', 'cup', 'decilitre', 'fluid ounce', 'gallon', 'imperial gallon', 'litre', 'microlitre', 'millilitre', 'pint', 'quart'}
    }
    
    valid_units = entity_unit_map.get(entity_name, set())
    
    if not valid_units:
        return ""  
    entity_pattern = re.compile(rf'{entity_name}.*?(\d+[\.\d+]*)\s*(\w+)', re.IGNORECASE)
    match = entity_pattern.search(extracted_text)
    
    if not match:
        return ""  
    value, unit = match.groups()
    unit = unit.lower()
    
    if unit not in valid_units:
        return ""  
    return f"{value} {unit}"

def predictor(image_link, category_id, entity_name):
    try:
        image = download_image(image_link)
        extracted_text = extract_text_from_image(image)
        processed_text = process_text(entity_name, extracted_text)
        print([(category_id),(entity_name),(processed_text)])
        return processed_text
    except Exception as e:
        print(f"Error processing {image_link}: {e}")
        return ""

def parallel_predictor(row):
    return predictor(row['image_link'], row['group_id'], row['entity_name'])

if __name__ == "__main__":
    DATASET_FOLDER = 'dataset/'
    test = pd.read_csv(os.path.join(DATASET_FOLDER, 'test.csv'))
    num_workers = min(cpu_count(), 14) 
    with Pool(num_workers) as pool:
        test['prediction'] = pool.map(parallel_predictor, [row for _, row in test.iterrows()])
    output_filename = os.path.join(DATASET_FOLDER, 'test_out.csv')
    test[['index', 'prediction']].to_csv(output_filename, index=False)
