import pandas as pd
import os

# Define paths
csv_path = 'us-housing-dataset/us-housing-dataset.csv'
output_path = 'us-housing-dataset/ca_2022_sold.pkl'

print(f"Reading {csv_path}...")
# Load necessary columns
cols = ['status', 'price', 'bed', 'bath', 'acre_lot', 'city', 'state', 'zip_code', 'house_size', 'prev_sold_date']
df = pd.read_csv(csv_path, usecols=cols)

print("Filtering for California houses sold in 2022...")
# Convert to datetime and filter
df['sold_date'] = pd.to_datetime(df['prev_sold_date'], errors='coerce')
ca_2022 = df[(df['state'] == 'California') & (df['sold_date'].dt.year == 2022)].copy()

# Drop the helper date column if not needed, or keep it
# ca_2022 = ca_2022.drop(columns=['sold_date'])

print(f"Found {len(ca_2022)} records.")

print(f"Saving to {output_path}...")
ca_2022.to_pickle(output_path)

print("Done!")
