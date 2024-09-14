import os
import pandas as pd
from tqdm import tqdm
import re
import pandas as pd
import warnings
from fuzzywuzzy import fuzz
import numpy as np
from tqdm import tqdm  
warnings.filterwarnings("ignore")

def clean_text(text):
    # Convert to lowercase and strip unwanted spaces
    text = text.lower().strip()
    # Remove unwanted characters like multiple spaces, non-alphanumeric except units
    text = re.sub(r'[^\w\s.,/]', ' ', text)
    text = re.sub(r'\s+', ' ', text)  # Normalize spaces
    return text

def remove(value):
    if value.endswith('.'):
        return value[:-1]
    return value

def process_text(entity_name, extracted_text):
    extracted_text = clean_text(str(extracted_text))
    
    # Define allowed units
    ALLOWED_UNITS = {
        'width': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
        'depth': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
        'height': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
        'item_weight': {'gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton'},
        'maximum_weight_recommendation': {'gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton'},
        'voltage': {'kilovolt', 'millivolt', 'volt'},
        'wattage': {'kilowatt', 'watt'},
        'item_volume': {
            'centilitre', 'cubic foot', 'cubic inch', 'cup', 'decilitre', 'fluid ounce',
            'gallon', 'imperial gallon', 'litre', 'microlitre', 'millilitre', 'pint', 'quart'
        }
    }

    # Handle abbreviations
    UNIT_CORRECTIONS = {
        'g': 'gram', 'gr': 'gram', 'gm': 'gram', 'kg': 'kilogram', 'kgs': 'kilogram', 
        'mg': 'milligram', 'mgs': 'milligram', 'μg': 'microgram', 'mcg': 'microgram', 
        'oz': 'ounce', 'lb': 'pound', 'lbs': 'pound', 'ton': 'ton',
        'ml': 'millilitre', 'l': 'litre', 'cl': 'centilitre', 'dl': 'decilitre',
        '"': 'inch', 'cm': 'centimetre', 'mm': 'millimetre', 'm': 'metre', 'ft': 'foot', 
        'yd': 'yard', 'v': 'volt', 'w': 'watt', 'kw': 'kilowatt', 'sq ft': 'cubic foot'
    }

    # Replace abbreviations with full units
    for abbr, full_unit in UNIT_CORRECTIONS.items():
        extracted_text = re.sub(r'\b' + re.escape(abbr) + r'\b', full_unit, extracted_text)

    # Extract numbers and units
    numbers = re.findall(r'\d+\.?\d*', extracted_text)
    units = re.findall(r'\b(?:' + '|'.join(ALLOWED_UNITS[entity_name]) + r')\b', extracted_text)

    # Handle multiple numbers (choose the number closest to a unit)
    if numbers and units:
        # Iterate through and match the closest number to each unit
        for unit in units:
            for number in numbers:
                # Assuming we take the first relevant number-unit pair
                return f"{remove(number)} {unit}"

    # Fuzzy matching for units if not directly matched
    extracted_words = re.findall(r'\b\w+\b', extracted_text)
    for word in extracted_words:
        best_match, best_score = "", 0
        for allowed_unit in ALLOWED_UNITS[entity_name]:
            score = fuzz.partial_ratio(word, allowed_unit)
            if score > best_score and score > 70:  # Improved threshold for better accuracy
                best_match, best_score = allowed_unit, score
        if best_match:
            return f"{remove(numbers[0])} {best_match}" if numbers else ""
    
    return ""

def process_image(row):
    index = row['index']
    entity_name = row['entity_name']
    prediction = row['prediction']
    processed_text = process_text(entity_name, prediction)
    return (index, processed_text)

if __name__ == "__main__":
    test = pd.read_csv('test_out.csv')
    
    results = []
    with tqdm(total=len(test), desc="Postprocessing images") as pbar:
        for _, row in test.iterrows():
            result = process_image(row)
            if result:
                results.append(result)
            pbar.update(1)
    
    results_df = pd.DataFrame(results, columns=['index', 'prediction'])
    output_filename = 'final_out.csv'
    results_df.to_csv(output_filename, index=False)
    print("Processing complete. Results saved to", output_filename)
