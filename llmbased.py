import os
import requests
from PIL import Image
from io import BytesIO
import pytesseract
import re
import pandas as pd
from PIL import Image
import torch
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True, torch_dtype=torch.float16)
model = model.to(device='cuda')

tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True)
model.eval()


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
        image = Image.open(image).convert('RGB')
        question = f'What is {entity_name} in the image?, reply with units aswell'
        msgs = [{'role': 'user', 'content': question}]
        res = model.chat(
            image=image,
            msgs=msgs,
            tokenizer=tokenizer,
            sampling=True, 
            temperature=0.7,
      
        )
        print([(category_id),(entity_name),(res['content'])])

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
