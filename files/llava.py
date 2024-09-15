
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
from langchain_ollama import OllamaLLM


model = OllamaLLM(model="gemma2:2b", temperature=0.1, max_tokens=20)


def process_text(entity_name, extracted_text):
    extracted_text=str(extracted_text)
    result=model.invoke(input=f'What is the {entity_name} in {extracted_text}?, if it exists then reply with answer else reply with "No answer"')
    print(result)
    return ""

def process_image(row):
    index = row['index']
    entity_name = row['entity_name']
    prediction = row['prediction']
    if prediction:
        processed_text = process_text(entity_name, prediction)
        return (index, processed_text)
    else:
        return (index,"")

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
    output_filename = 'crazy.csv'
    results_df.to_csv(output_filename, index=False)
    print("Processing complete. Results saved to", output_filename)