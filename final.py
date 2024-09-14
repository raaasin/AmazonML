
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

def remove(value):
    if value.endswith('.'):
        return value[:-1]
    return value
def process_text(entity_name, extracted_text):
    extracted_text=str(extracted_text)
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
    
    results_df = pd.DataFrame(results, columns=['index', 'entity_value'])
    output_filename = 'final_out.csv'
    results_df.to_csv(output_filename, index=False)
    print("Processing complete. Results saved to", output_filename)