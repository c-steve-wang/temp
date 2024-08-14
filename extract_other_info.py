import os
import pandas as pd
import sqlite3
from tqdm import tqdm

# Directory containing CSV files
csv_directory = 'data/2023-02-11/twitter-cmdresponses/'
cell_file_path = 'tr_death_id_02_11.csv'
# SQLite database file
db_file = 'example.db'

# Create a new SQLite database (or connect to an existing one)
conn = sqlite3.connect(db_file)
cursor = conn.cursor()

# Create a table in the SQLite database with the new columns
cursor.execute('''
CREATE TABLE IF NOT EXISTS records (
    title TEXT,
    author_id INTEGER,
    created_at TEXT,
    text TEXT,
    lang TEXT,
    source TEXT,
    public_metrics_like_count INTEGER,
    public_metrics_quote_count INTEGER,
    public_metrics_reply_count INTEGER,
    entities_annotations TEXT,
    entities_cashtags TEXT,
    entities_hashtags TEXT,
    entities_mentions TEXT,
    entities_urls TEXT,
    author_created_at TEXT,
    author_username TEXT,
    author_name TEXT,
    author_description TEXT,
    author_entities_description_cashtags TEXT,
    author_entities_description_urls TEXT,
    author_entities_url_urls TEXT,
    author_location TEXT,
    author_profile_image_url TEXT,
    author_protected BOOLEAN,
    author_public_metrics_followers_count INTEGER,
    author_public_metrics_following_count INTEGER,
    author_public_metrics_listed_count INTEGER,
    author_url TEXT,
    author_verified BOOLEAN,
    geo_coordinates_coordinates TEXT,
    geo_place_id TEXT,
    id INTEGER PRIMARY KEY,
    referenced_tweets_retweeted_id INTEGER,
    public_metrics_retweet_count INTEGER,
    possibly_sensitive BOOLEAN,
    author_public_metrics_tweet_count INTEGER,
    geo_place_type TEXT,
    geo_coordinates_type TEXT,
    geo_country TEXT,
    geo_country_code TEXT,
    geo_full_name TEXT,
    geo_geo_bbox TEXT,
    geo_geo_type TEXT,
    geo_id TEXT,
    geo_name TEXT
)
''')

# Function to insert or update records
def insert_or_update(df, conn):
    for index, row in df.iterrows():
        try:
            data = tuple(row)
            placeholders = ', '.join(['?'] * len(row))
            update_columns = ', '.join([f"{col}=excluded.{col}" for col in df.columns if col != 'id'])
            sql = f'''
            INSERT INTO records ({', '.join(df.columns)}) VALUES ({placeholders})
            ON CONFLICT(id) DO UPDATE SET
            {update_columns}
            '''
            conn.execute(sql, data)
        except:
            continue  # Skip the problematic row
    conn.commit()

# Read each CSV file and insert/update its contents into the database
files = [f for f in os.listdir(csv_directory) if f.endswith('.csv') and 'tmp' not in f]
files_read = 0

for csv_file in tqdm(files, desc="Processing CSV files"):
    try:
        file_path = os.path.join(csv_directory, csv_file)
        df = pd.read_csv(file_path, dtype=str, low_memory=False)  # Read all columns as strings to avoid mixed types
        df.drop(columns=['Unnamed: 0', 'referenced_tweets_replied_to_id'], errors='ignore', inplace=True)  # Drop any unwanted index columns and the problematic column

        # Rename columns to replace periods with underscores
        df.columns = [col.replace('.', '_') for col in df.columns]

        # Insert or update the records in the database
        insert_or_update(df, conn)
        files_read += 1
    except:
        continue  # Skip the problematic file

print(f"Number of files read: {files_read}")

# Function to update records based on id and author_id
def update_csv_with_db_values(cell_file_path):
    try:
        cells_df = pd.read_csv(cell_file_path)
        cells_df.columns = [col.replace('.', '_') for col in cells_df.columns]  # Ensure column names match the database

        updated_rows = []
        for _, row in tqdm(cells_df.iterrows(), total=cells_df.shape[0], desc="Updating CSV with DB values"):
            id_value = row['id']
            author_id_value = row['author_id']
            query = '''
            SELECT 
                public_metrics_like_count, author_url, author_verified,
                lang, author_public_metrics_followers_count, author_public_metrics_following_count,
                author_public_metrics_listed_count, geo_coordinates_type, geo_country,
                geo_country_code, geo_full_name
            FROM records WHERE id = ? AND author_id = ?
            '''
            cursor.execute(query, (id_value, author_id_value))
            result = cursor.fetchone()
            if result:
                (like_count, author_url, author_verified, lang, 
                 followers_count, following_count, listed_count, 
                 coordinates_type, country, country_code, full_name) = result
                row['public_metrics_like_count'] = like_count
                row['author_url'] = author_url
                row['author_verified'] = author_verified
                row['lang'] = lang
                row['author_public_metrics_followers_count'] = followers_count
                row['author_public_metrics_following_count'] = following_count
                row['author_public_metrics_listed_count'] = listed_count
                row['geo_coordinates_type'] = coordinates_type
                row['geo_country'] = country
                row['geo_country_code'] = country_code
                row['geo_full_name'] = full_name
            updated_rows.append(row)

        updated_df = pd.DataFrame(updated_rows)
        updated_df.to_csv(cell_file_path, index=False)
    except:
        pass  # Skip the problematic file

# Path to the CSV file containing id and author_id

# Update the CSV file based on the cell file
update_csv_with_db_values(cell_file_path)

# Close the database connection
conn.close()
