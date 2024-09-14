import pandas as pd

# Read the final_out.csv file
final_out_df = pd.read_csv('final_out.csv')

# Count null values in the "prediction" column of final_out.csv
final_out_null_count = final_out_df['prediction'].isnull().sum()

# Read the test_count.csv file
test_count_df = pd.read_csv('test_out.csv')

# Count null values in the "prediction" column of test_count.csv
test_count_null_count = test_count_df['prediction'].isnull().sum()

per1=(131287-test_count_null_count)/131287
per2=(131287-final_out_null_count)/131287
print(f"After OCR we got {per1}%, total count is {131287-test_count_null_count}")
print(f"After postprocessing we got {per2}%, total count is {131287-final_out_null_count}")