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
from PIL import ImageEnhance
from textblob import TextBlob
warnings.filterwarnings("ignore")
print(torch.cuda.is_available())

reader = easyocr.Reader(['en'], gpu=True)

def download_image(image_url):
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    return img

def preprocess_image(image):
    image_np = np.array(image)
    if len(image_np.shape) == 3 and image_np.shape[2] == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    elif len(image_np.shape) == 2:
        gray = image_np
    else:
        raise ValueError("Unsupported image format")
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return binary

def correct_spelling(text):
    blob = TextBlob(text)
    return str(blob.correct())

def extract_text_from_image(image):
    processed_image = preprocess_image(image)
    processed_image_np = np.array(processed_image)
    text = reader.readtext(processed_image_np, detail=0, text_threshold=0.6)
    extracted_text = ' '.join(text)
    corrected_text = correct_spelling(extracted_text)
    
    return corrected_text

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



def predictor(image_link, category_id, entity_name):
    try:
        image = download_image(image_link)
        extracted_text = extract_text_from_image(image)
        processed_text = process_text(entity_name, extracted_text)
        print([(category_id), (entity_name), (processed_text)])
        return processed_text
    except Exception as e:
        print(f"Error processing {image_link}: {e}")
        return ""

if __name__ == "__main__":
    DATASET_FOLDER = 'dataset/'
    test = pd.read_csv(os.path.join(DATASET_FOLDER, 'test.csv'))
    test['prediction'] = test.apply(lambda row: predictor(row['image_link'], row['group_id'], row['entity_name']), axis=1)
    output_filename = os.path.join(DATASET_FOLDER, 'test_out.csv')
    test[['index', 'prediction']].to_csv(output_filename, index=False)
