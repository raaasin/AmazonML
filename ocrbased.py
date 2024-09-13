import os
import requests
from PIL import Image
from io import BytesIO
import pytesseract
import re
import pandas as pd

pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

def download_image(image_url):
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    return img

def extract_text_from_image(image):
    text = pytesseract.image_to_string(image)
    return text
def process_text(entity_name, extracted_text):
    '''
    Function to process the extracted text and find the relevant entity value.
    '''
    entity_unit_map = {
    'width': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'depth': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'height': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'item_weight': {'gram',
        'kilogram',
        'microgram',
        'milligram',
        'ounce',
        'pound',
        'ton'},
    'maximum_weight_recommendation': {'gram',
        'kilogram',
        'microgram',
        'milligram',
        'ounce',
        'pound',
        'ton'},
    'voltage': {'kilovolt', 'millivolt', 'volt'},
    'wattage': {'kilowatt', 'watt'},
    'item_volume': {'centilitre',
        'cubic foot',
        'cubic inch',
        'cup',
        'decilitre',
        'fluid ounce',
        'gallon',
        'imperial gallon',
        'litre',
        'microlitre',
        'millilitre',
        'pint',
        'quart'}
    }

    return extracted_text
def predictor(image_link, category_id, entity_name):
    global count
    '''
    Function to download the image, extract text using OCR, and find the relevant entity value.
    '''
    try:
        image = download_image(image_link)
        extracted_text = extract_text_from_image(image)
        processed_text = (entity_name, extracted_text)
        print([(category_id),(entity_name),(extracted_text)])
        #return processed_text
    except Exception as e:
        print(f"Error processing {image_link}: {e}")
        return ""

if __name__ == "__main__":
    DATASET_FOLDER = 'dataset/'
    
    test = pd.read_csv(os.path.join(DATASET_FOLDER, 'test.csv'))
    
    test['prediction'] = test.apply(
        lambda row: predictor(row['image_link'], row['group_id'], row['entity_name']), axis=1)
    
    output_filename = os.path.join(DATASET_FOLDER, 'test_out.csv')
    test[['index', 'prediction']].to_csv(output_filename, index=False)
