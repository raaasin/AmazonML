import os
import requests
from PIL import Image
from io import BytesIO
import pytesseract
import re
import pandas as pd
from transformers import AutoProcessor, AutoModelForCausalLM
from huggingface_hub import hf_hub_download
from PIL import Image
import torch


processor = AutoProcessor.from_pretrained("microsoft/git-base-textvqa")
model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-textvqa")

count = 0
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

def download_image(image_url):
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    return img

def extract_text_from_image(image):
    text = pytesseract.image_to_string(image)
    return text

def predictor(image_link, category_id, entity_name):
    global count
    try:
        image = download_image(image_link)
        pixel_values = processor(images=image, return_tensors="pt").pixel_values
        question = f"what is the {entity_name} in the image?"
        input_ids = processor(text=question, add_special_tokens=False).input_ids
        input_ids = [processor.tokenizer.cls_token_id] + input_ids
        input_ids = torch.tensor(input_ids).unsqueeze(0)
        generated_ids = model.generate(pixel_values=pixel_values, input_ids=input_ids, max_length=50)
        print(processor.batch_decode(generated_ids, skip_special_tokens=True))

        count+=1
        return ""
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
