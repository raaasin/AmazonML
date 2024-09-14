import os
import requests
from PIL import Image
from io import BytesIO
import easyocr
import re
import pandas as pd
import torch
import warnings
from fuzzywuzzy import fuzz
import cv2
import numpy as np
from tqdm import tqdm  
import signal
import sys
from multiprocessing import Pool 
warnings.filterwarnings("ignore")
print(torch.cuda.is_available())

reader = easyocr.Reader(['en'], gpu=True)

def download_image(image_url):
    try:
        response = requests.get(image_url)  # Set a timeout to avoid long waits on bad connections
        img = Image.open(BytesIO(response.content))
        return img
    except Exception as e:
        print(f"Failed to download image: {e}")
        return None

def preprocess_image(image):
    """Optimized image preprocessing using faster techniques."""
    image_np = np.array(image)
    if len(image_np.shape) == 3 and image_np.shape[2] == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    elif len(image_np.shape) == 2:
        gray = image_np
    else:
        raise ValueError("Unsupported image format")

    # Use faster denoising and thresholding techniques
    gray = cv2.fastNlMeansDenoising(gray, None, h=7, templateWindowSize=7)  # Denoising for clearer text
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # Fast adaptive thresholding

    return binary


def extract_text_from_image(image):
    processed_image = preprocess_image(image)
    processed_image_np = np.array(processed_image)
    text = reader.readtext(processed_image_np, detail=0)
    extracted_text = ' '.join(text)
    return extracted_text

def remove(value):
    if value.endswith('.'):
        return value[:-1]
    return value
def process_text(entity_name, extracted_text):

    ALLOWED_UNITS = {
    'width': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'depth': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'height': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'item_weight': {
        'gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton'
    },
    'maximum_weight_recommendation': {
        'gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton'
    },
    'voltage': {'kilovolt', 'millivolt', 'volt'},
    'wattage': {'kilowatt', 'watt'},
    'item_volume': {
        'centilitre', 'cubic foot', 'cubic inch', 'cup', 'decilitre', 'fluid ounce',
        'gallon', 'imperial gallon', 'litre', 'microlitre', 'millilitre', 'pint', 'quart'
    }
}
    

    UNIT_CORRECTIONS = UNIT_CORRECTIONS = {
    'g': 'gram',
    'gr': 'gram',
    'gm': 'gram',
    'kg': 'kilogram',
    'kgs': 'kilogram',
    'mg': 'milligram',
    'mgs': 'milligram',
    'μg': 'microgram',  
    'mcg': 'microgram', 
    'oz': 'ounce',
    'lb': 'pound',
    'lbs': 'pound',
    'ton': 'ton',
    'ml': 'millilitre',
    'mL': 'millilitre',
    'microl': 'microlitre',
    'litre': 'litre',
    'liter': 'litre', 
    'l': 'litre',
    'ltr': 'litre',
    'cl': 'centilitre',
    'dl': 'decilitre',
    'fl oz': 'fluid ounce',
    'fluid ounce': 'fluid ounce',
    'gal': 'gallon',
    'imperial gallon': 'imperial gallon',
    'cup': 'cup',
    'pint': 'pint',
    'qt': 'quart',
    'quart': 'quart',
    'cubic foot': 'cubic foot',
    'cubic inch': 'cubic inch',
    '"': 'inch',
    'inches': 'inch',
    'in': 'inch',
    'cm': 'centimetre',
    'centimetre': 'centimetre',
    'centimeter': 'centimetre',
    'mm': 'millimetre',
    'millimetre': 'millimetre',
    'millimeter': 'millimetre',
    'm': 'metre',
    'meter': 'metre',
    'metre': 'metre',
    'ft': 'foot',
    'feet': 'foot',
    'yd': 'yard',
    'yards': 'yard',
    'v': 'volt',
    'volt': 'volt',
    'kv': 'kilovolt',
    'mv': 'millivolt',
    'w': 'watt',
    'watt': 'watt',
    'kw': 'kilowatt',
    'sq ft': 'cubic foot',  
    'sq in': 'cubic inch',
}

    for abbr, full_unit in UNIT_CORRECTIONS.items():
        extracted_text = re.sub(r'\b' + abbr + r'\b', full_unit, extracted_text)
    numbers = re.findall(r'\d+\.?\d*', extracted_text)

    units = re.findall(r'\b(?:' + '|'.join(ALLOWED_UNITS[entity_name]) + r')\b', extracted_text.lower())

    if numbers and units:
        value = numbers[0]
        unit = units[0]
        return f"{remove(value)} {unit}"
    extracted_words = re.findall(r'\b\w+\b', extracted_text.lower())
    for word in extracted_words:
        best_match, best_score = "", 0
        for allowed_unit in ALLOWED_UNITS[entity_name]:
            score = fuzz.ratio(word, allowed_unit)
            if score > best_score and score > 50: 
                best_match, best_score = allowed_unit, score
        if best_match:
            return f"{remove(numbers[0])} {best_match}" if numbers else ""
    return ""



def predictor(row_tuple):
    index, row = row_tuple  # Unpack the tuple
    image_link, category_id, entity_name = row['image_link'], row['group_id'], row['entity_name']
    try:
        image = download_image(image_link)
        extracted_text = extract_text_from_image(image)
        processed_text = process_text(entity_name, extracted_text)
        print(f"{index}, {image_link}, {entity_name}, {processed_text}")
        return processed_text
    except Exception as e:
        print(f"Error processing {image_link}: {e}")
        return ""

def save_progress(df, output_filename):
    df.to_csv(output_filename, index=False)
    print(f"Progress saved to {output_filename}")



if __name__ == "__main__":
    DATASET_FOLDER = 'dataset/'
    output_filename = os.path.join(DATASET_FOLDER, 'test_out.csv')
    test = pd.read_csv(os.path.join(DATASET_FOLDER, 'test.csv'))

    with Pool(processes=2) as pool:  # Adjust `processes` based on the number of CPU cores
        test['prediction'] = list(tqdm(pool.imap(predictor, test.iterrows()), total=len(test)))

    save_progress(test[['index', 'prediction']], output_filename)