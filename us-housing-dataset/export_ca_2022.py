import pandas as pd
import requests

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

print(f"Found {len(ca_2022)} records.")

# --- CENSUS DATA ENRICHMENT ---
print("Fetching Census ACS data for ZIP codes...")

# ACS 5-year endpoint (latest available for 2022)
BASE_URL = "https://api.census.gov/data/2022/acs/acs5"

variables = {
    "median_income": "B19013_001E",
    "population": "B01003_001E",
    "poverty": "B17001_002E",        # below poverty
    "total_poverty_universe": "B17001_001E",
    "education_total": "B15003_001E",
    "bachelors_plus": "B15003_022E"  # simplification (bachelor+ approx)
}
var_list = ",".join(variables.values())

params = {
    "get": f"NAME,{var_list}",
    "for": "zip code tabulation area:*"
}

r = requests.get(BASE_URL, params=params)
if r.status_code == 200:
    data = r.json()
    header = data[0]
    rows = data[1:]
    
    # Create a dataframe from the API response
    census_df = pd.DataFrame(rows, columns=header)
    
    # Convert 'zip code tabulation area' to numeric to match the 'zip_code' column in our housing df
    census_df['zip_code'] = pd.to_numeric(census_df['zip code tabulation area'], errors='coerce')
    
    # The housing df zip_code is also numeric (float)
    ca_2022['zip_code'] = pd.to_numeric(ca_2022['zip_code'], errors='coerce')
    
    # Ensure types for calculation
    for var in variables.values():
        census_df[var] = pd.to_numeric(census_df[var], errors='coerce')

    # Replace Census sentinel values with NaN.
    # The Census API uses -666666666 to indicate suppressed/unavailable data.
    # Any negative value in these count/income columns is invalid.
    for var in variables.values():
        census_df[var] = census_df[var].where(census_df[var] >= 0, other=float('nan'))

    # Calculate derived rates (NaN propagates if denominator or numerator is NaN)
    census_df['poverty_rate'] = census_df['B17001_002E'] / census_df['B17001_001E']
    census_df['bachelors_rate'] = census_df['B15003_022E'] / census_df['B15003_001E']

    # Also replace any 0-denominator results (division by zero -> inf/nan)
    census_df['poverty_rate'] = census_df['poverty_rate'].replace([float('inf'), float('-inf')], float('nan'))
    census_df['bachelors_rate'] = census_df['bachelors_rate'].replace([float('inf'), float('-inf')], float('nan'))

    # Rename raw variables to readable names
    rename_dict = {v: k for k, v in variables.items()}
    census_df = census_df.rename(columns=rename_dict)

    # Keep only the final enrichment columns
    census_cols = ['zip_code', 'median_income', 'population', 'poverty_rate', 'bachelors_rate']
    census_df = census_df[census_cols]

    # Merge with the CA 2022 dataset
    print("Merging Census data with housing data...")
    initial_count = len(ca_2022)
    ca_2022 = ca_2022.merge(census_df, on='zip_code', how='left')

    # Drop rows with any missing census value (sentinel-replaced NaN or unmatched ZIP)
    print("Dropping rows with missing or invalid Census data:")
    for col in ['median_income', 'population', 'poverty_rate', 'bachelors_rate']:
        n_missing = ca_2022[col].isna().sum()
        print(f"  {col}: {n_missing} missing")

    ca_2022 = ca_2022.dropna(subset=['median_income', 'population', 'poverty_rate', 'bachelors_rate'])
    final_count = len(ca_2022)
    dropped_count = initial_count - final_count

    print(f"Successfully enriched data.")
    print(f"Rows dropped due to missing/invalid Census information: {dropped_count}")
    print(f"Final record count: {final_count}")
else:
    print(f"Failed to fetch Census data. Status code: {r.status_code}")
    print(r.text)

# --- END CENSUS DATA ENRICHMENT ---

# --- SCHOOL & LIBRARY ENRICHMENT ---
# schools_with_zipcode.csv / library_with_zipcode.csv columns:
#   longitude, latitude, name, amenity, zipcode (int64), zip_distance_km (float64)
# We aggregate to ZIP-level summary features, then left-join so no housing rows
# are dropped for ZIPs with no school or library match.

print("Loading and aggregating school data...")
schools_raw = pd.read_csv('us-housing-dataset/schools_with_zipcode.csv')
school_zip = (
    schools_raw
    .groupby('zipcode')
    .agg(
        school_count=('name', 'count'),
        avg_school_dist_km=('zip_distance_km', 'mean'),
        min_school_dist_km=('zip_distance_km', 'min'),
    )
    .reset_index()
    .rename(columns={'zipcode': 'zip_code'})
)
school_zip['zip_code'] = school_zip['zip_code'].astype(float)
print(f"  {len(schools_raw):,} school records across {len(school_zip):,} ZIP codes.")

print("Loading and aggregating library data...")
libs_raw = pd.read_csv('us-housing-dataset/library_with_zipcode.csv')
lib_zip = (
    libs_raw
    .groupby('zipcode')
    .agg(
        library_count=('name', 'count'),
        avg_library_dist_km=('zip_distance_km', 'mean'),
        min_library_dist_km=('zip_distance_km', 'min'),
    )
    .reset_index()
    .rename(columns={'zipcode': 'zip_code'})
)
lib_zip['zip_code'] = lib_zip['zip_code'].astype(float)
print(f"  {len(libs_raw):,} library records across {len(lib_zip):,} ZIP codes.")

_before = len(ca_2022)
ca_2022 = ca_2022.merge(school_zip, on='zip_code', how='left')
ca_2022 = ca_2022.merge(lib_zip,    on='zip_code', how='left')
assert len(ca_2022) == _before, "Row count changed unexpectedly after school/library merge!"

print(f"  Rows with no school match (NaN kept): {ca_2022['school_count'].isna().sum():,}")
print(f"  Rows with no library match (NaN kept): {ca_2022['library_count'].isna().sum():,}")
print("School & library enrichment complete.")
print(f"Final columns: {ca_2022.columns.tolist()}")

# --- END SCHOOL & LIBRARY ENRICHMENT ---

print(f"Saving to {output_path}...")
ca_2022.to_pickle(output_path)

print("Done!")
