# Ctrl + Alt + Return - Documentation

## Overview

This project involves processing images and extracting relevant information, specifically units and values, from the images' text. It uses a combination of machine learning models for text generation and natural language processing to clean up the extracted data.

### Libraries and Dependencies

You need the following libraries to run the code:

- `transformers`: For loading and using the pretrained PaliGemma model from Hugging Face.
- `torch`: For model inference.
- `PIL`: For handling image operations.
- `requests`: For fetching images from URLs.
- `pandas`: For handling data in tabular form.
- `tqdm`: For progress bars during processing.
- `fuzzywuzzy`: For fuzzy string matching.
- `re`: For regular expression operations.
- `numpy`: For array handling.

## Part 1: Image Processing Using a Pretrained Model

1. **Model Setup:**

   - The script uses the Hugging Face library to load a model called `google/paligemma-3b-mix-224` from the Hugging Face Hub.
   - A processor (`AutoProcessor`) is used to format the input text and images, and the model itself is used for generating conditional output based on the input image and a text prompt.
2. **Process Image:**

   - For each image, a text prompt is created using the `entity_name` (e.g., "What is the height?" or "What is the weight?").
   - The image is fetched using its URL, and the model generates an output that is interpreted as the predicted information (e.g., height, weight, etc.).
3. **Main Script Workflow:**

   - The script loads a CSV file (`test.csv`) containing the image URLs and entity names.
   - For each row, the image is processed to predict the relevant information.
   - Results are saved to a CSV file (`final_out.csv`), with the image index and the predicted text.

## Part 2: Postprocessing Predictions

Once the model outputs are generated, they often need cleaning up and validation, particularly in matching the correct units to the values. This is done in the second part of the code.

1. **Cleaning Units:**

   - The `process_text()` function takes the model's predicted output (e.g., "10 cm") and checks if it contains a valid unit.
   - A dictionary (`ALLOWED_UNITS`) defines the correct units for various entities (like weight, height, etc.).
   - Another dictionary (`UNIT_CORRECTIONS`) helps map abbreviations or misspelled units (e.g., `cm` to `centimetre`).
   - The script uses regular expressions to search for numbers and units, ensuring the extracted values are valid.
2. **Fuzzy Matching:**

   - If no valid unit is found in the text, the `fuzzywuzzy` library is used to find approximate matches between the extracted text and the allowed units.
   - If a match is found, it replaces the detected text with the correct unit.
3. **Postprocessing Workflow:**

   - The script loads the output from the first part (`test_out.csv`) and applies the unit cleaning process.
   - The results are saved again in a cleaned-up format, ready for final use in `final_out.csv`.

## How to Run the Code

1. Install dependencies:

   ```bash
   pip install transformers accelerate pandas tqdm fuzzywuzzy Pillow torch
   ```
2. Run the script:

   ```bash
   python model.py
   python recheck.py
   ```

Make sure you have the `test.csv` and `test_out.csv` files in the same directory. The final output will be saved to `final_out.csv`.
