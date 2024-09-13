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
warnings.filterwarnings("ignore")

print(torch.cuda.is_available())
reader = easyocr.Reader(['en'], gpu=True)

def download_image(image_url):
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    return img

def extract_text_from_image(image):
    text = reader.readtext(image, detail=0, text_threshold=0.1)
    return ' '.join(text)

from fuzzywuzzy import fuzz

from fuzzywuzzy import fuzz

def process_text(entity_name, extracted_text):
    # Define allowed units for each entity type with common abbreviations
    ALLOWED_UNITS = {
        'item_weight': ['gram', 'kilogram', 'milligram', 'ounce', 'pound'],
        'item_volume': ['milliliter', 'liter', 'cup', 'fluid ounce', 'gallon'],
        'height': ['centimetre', 'inch', 'millimetre', 'metre'],
        'width': ['centimetre', 'inch', 'millimetre', 'metre'],
        'depth': ['centimetre', 'inch', 'millimetre', 'metre'],
    }
    
    # Define abbreviations and symbols to their corresponding full unit names
    UNIT_CORRECTIONS = {
        'g': 'gram',
        'kg': 'kilogram',
        'mg': 'milligram',
        'oz': 'ounce',
        'lb': 'pound',
        'ml': 'milliliter',
        'l': 'liter',
        'ltr': 'liter',
        'fl oz': 'fluid ounce',
        '"': 'inch',
        '""': 'inch',
        'cm': 'centimetre',
        'mm': 'millimetre',
        'm': 'metre',
    }
    
    # Replace common abbreviations and symbols in the extracted text
    for abbr, full_unit in UNIT_CORRECTIONS.items():
        extracted_text = re.sub(r'\b' + abbr + r'\b', full_unit, extracted_text)

    # Extract numbers from text
    numbers = re.findall(r'\d+\.?\d*', extracted_text)
    
    # Check if any allowed unit for the entity is present in the extracted text
    units = re.findall(r'\b(?:' + '|'.join(ALLOWED_UNITS[entity_name]) + r')\b', extracted_text.lower())

    # If exact unit matches are found
    if numbers and units:
        value = numbers[0]
        unit = units[0]
        return f"{value} {unit}"
    
    # If no exact match for the unit, try fuzzy matching
    extracted_words = re.findall(r'\b\w+\b', extracted_text.lower())
    for word in extracted_words:
        best_match, best_score = "", 0
        for allowed_unit in ALLOWED_UNITS[entity_name]:
            score = fuzz.ratio(word, allowed_unit)
            if score > best_score and score > 75:  # Threshold for similarity
                best_match, best_score = allowed_unit, score
        
        # If a fuzzy match is found for the unit
        if best_match:
            return f"{numbers[0]} {best_match}" if numbers else ""

    # If no match, return empty
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
