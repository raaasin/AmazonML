import os
from PIL import Image
import easyocr
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
import warnings
import requests
warnings.filterwarnings("ignore")
count = 0
print(torch.cuda.is_available())
reader = easyocr.Reader(['en'], gpu=True)


def open_image(image_link):
    try:
        img = Image.open(requests.get(image_link, stream=True).raw).convert('RGB')  # Ensure image is in RGB format
        return img
    except Exception as e:
        print(f"Error opening image {image_link}: {e}")
        return None

def extract_text_from_image(image):
    try:
        image_np = np.array(image)  
        text = reader.readtext(image_np, detail=0, text_threshold=0.4)
        return ' '.join(text)
    except Exception as e:
        print(f"Error extracting text: {e}")
        return ""

def process_image(row):
    global count
    index = count
    count += 1
    image_link = row['image_link']
    group_id = row['group_id']
    entity_name = row['entity_name']
    try:
        image = open_image(image_link)
        if image is not None:
            extracted_text = extract_text_from_image(image)
            return (index, entity_name, group_id, extracted_text)
        else:
            return (index, entity_name, group_id, "")
    except Exception as e:
        print(f"Error processing {image_link}: {e}")
        return (index, entity_name, group_id, "")

if __name__ == "__main__":
    test = pd.read_csv(os.path.join('small_train.csv'))
    
    results = []
    with tqdm(total=len(test), desc="Processing images") as pbar:
        for _, row in test.iterrows():
            result = process_image(row)
            if result:
                results.append(result)
            pbar.update(1)
    
    results_df = pd.DataFrame(results, columns=['index', 'entity_name', 'group_id', 'prediction'])
    output_filename = 'ocr.csv'
    results_df.to_csv(output_filename, index=False)
    print("Processing complete. Results saved to", output_filename)
