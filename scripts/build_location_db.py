import os
import sqlite3
import pandas as pd

# Define file paths
base_dir = os.path.dirname(__file__)
raw_dir = os.path.abspath(os.path.join(base_dir, "data", "raw"))
processed_dir = os.path.abspath(os.path.join(base_dir, "data", "processed"))
os.makedirs(processed_dir, exist_ok=True)

# Input CSVs
disease_path = os.path.join(raw_dir, "kenya_disease_county_matrix.csv")
xwalk_path = os.path.join(raw_dir, "sitecode_county_xwalk.csv")
rainy_path = os.path.join(raw_dir, "kenya_counties_rainy_seasons.csv")
who_path = os.path.join(raw_dir, "who_bulletin.csv")

# Output DB
db_path = os.path.join(processed_dir, "location_data.sqlite")

# Read CSVs
disease_df = pd.read_csv(disease_path)
xwalk_df = pd.read_csv(xwalk_path)
rainy_df = pd.read_csv(rainy_path)
who_df = pd.read_csv(who_path)

# Write to SQLite
conn = sqlite3.connect(db_path)
disease_df.to_sql('county_disease_info', conn, if_exists='replace', index=False)
xwalk_df.to_sql('sitecode_county_xwalk', conn, if_exists='replace', index=False)
rainy_df.to_sql('county_rainy_seasons', conn, if_exists='replace', index=False)
who_df.to_sql('who_bulletin', conn, if_exists='replace', index=False)

conn.commit()
conn.close()

print(f"SQLite database written to: {db_path}")
