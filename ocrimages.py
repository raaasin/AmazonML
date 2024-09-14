import os
from PIL import Image
import urllib.parse
import easyocr
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

print(torch.cuda.is_available())
reader = easyocr.Reader(['en'], gpu=True)

def get_image_filename(image_url):
    output_folder = 'images'
    path = urllib.parse.urlsplit(image_url).path
    filename = os.path.basename(path)
    return os.path.join(output_folder, filename)

def open_image(file_path):
    try:
        img = Image.open(file_path).convert('RGB')  # Ensure image is in RGB format
        img = img.resize((img.width // 2, img.height // 2))  # Resize for faster processing
        return img
    except Exception as e:
        print(f"Error opening image {file_path}: {e}")
        return None

def extract_text_from_image(image):
    try:
        image_np = np.array(image)  # Convert PIL image to numpy array
        # Adjust easyocr parameters for faster processing
        text = reader.readtext(image_np, detail=0, text_threshold=0.4)
        return ' '.join(text)
    except Exception as e:
        print(f"Error extracting text: {e}")
        return ""

def process_image(row):
    index = row['index']
    image_link = row['image_link']
    group_id = row['group_id']
    entity_name = row['entity_name']
    file_path = get_image_filename(image_link)
    try:
        image = open_image(file_path)
        if image is not None:
            extracted_text = extract_text_from_image(image)
            return (index, entity_name, group_id, extracted_text)
        else:
            return (index, entity_name, group_id, "")
    except Exception as e:
        print(f"Error processing {image_link}: {e}")
        return (index, entity_name, group_id, "")

if __name__ == "__main__":
    DATASET_FOLDER = 'dataset/'
    test = pd.read_csv(os.path.join(DATASET_FOLDER, 'test.csv'))
    
    results = []
    with tqdm(total=len(test), desc="Processing images") as pbar:
        for _, row in test.iterrows():
            result = process_image(row)
            if result:
                results.append(result)
            pbar.update(1)
    
    results_df = pd.DataFrame(results, columns=['index', 'entity_name', 'group_id', 'prediction'])
    output_filename = 'test_out.csv'
    results_df.to_csv(output_filename, index=False)
    print("Processing complete. Results saved to", output_filename)
