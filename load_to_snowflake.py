import argparse
import snowflake.connector
import pandas as pd
from snowflake.connector.pandas_tools import write_pandas
import ast

# Command-line arguments setup
def get_arguments():
    parser = argparse.ArgumentParser(description="Snowflake Operations Script")
    
    # Add options for creating and deleting resources
    parser.add_argument('--create_db', action='store_true', help="Create the database if it does not exist.")
    parser.add_argument('--create_schema', action='store_true', help="Create the schema if it does not exist.")
    parser.add_argument('--create_warehouse', action='store_true', help="Create the warehouse if it does not exist.")
    parser.add_argument('--create_table', action='store_true', help="Create the table if it does not exist.")
    parser.add_argument('--delete_table', action='store_true', help="Delete the table if it exists.")
    parser.add_argument('--csv_path', type=str, help="Path to the CSV file containing the data.")
    
    # Add the '--all' option
    parser.add_argument('--all', action='store_true', help="Perform all operations (create DB, schema, warehouse, table, delete table if exists, and load CSV).")
    
    return parser.parse_args()

# Configuration variables
SNOWFLAKE_USER = 'sanysandish'
SNOWFLAKE_PASSWORD = '123!@#POOJAsany'
SNOWFLAKE_ACCOUNT = 'FLB33416'
SNOWFLAKE_WAREHOUSE = 'usf_warehouse'
SNOWFLAKE_DATABASE = 'usf_pooja'
SNOWFLAKE_SCHEMA = 'tweets'
SNOWFLAKE_ROLE = 'ACCOUNTADMIN'

# SQL queries
CREATE_DATABASE_SQL = f"""
CREATE DATABASE IF NOT EXISTS {SNOWFLAKE_DATABASE};
"""

CREATE_SCHEMA_SQL = f"""
CREATE SCHEMA IF NOT EXISTS {SNOWFLAKE_DATABASE}.{SNOWFLAKE_SCHEMA};
"""

CREATE_WAREHOUSE_SQL = f"""
CREATE OR REPLACE WAREHOUSE {SNOWFLAKE_WAREHOUSE} 
WITH WAREHOUSE_SIZE = 'SMALL' 
AUTO_RESUME = TRUE 
AUTO_SUSPEND = 60;
"""

CREATE_TABLE_SQL = f"""
CREATE OR REPLACE TABLE {SNOWFLAKE_DATABASE}.{SNOWFLAKE_SCHEMA}.PREDICTED_TWEETS (
    AUTHOR STRING,
    CONTENT STRING,
    COUNTRY STRING,
    DATE_TIME STRING,
    ID INT,
    LANGUAGE STRING,
    LATITUDE FLOAT,
    LONGITUDE FLOAT,
    NUMBER_OF_LIKES INT,
    NUMBER_OF_SHARES INT,
    EMOTION STRING,
    HASHTAGS STRING,
    HASHTAGS_FINAL ARRAY,
    CONTENT_LENGTH INT,
    HASHTAGS_COUNT INT,
    POPULARITY FLOAT
);
"""


DELETE_TABLE_SQL = f"""
DROP TABLE IF EXISTS {SNOWFLAKE_DATABASE}.{SNOWFLAKE_SCHEMA}.PREDICTED_TWEETS;
"""

def clean_hashtags(hashtag_str):
    # Remove unnecessary characters like outer brackets and extra quotes
    cleaned_str = hashtag_str.strip("[ ]").replace("\"", "").replace("'", "")
    # Split the cleaned string into a list of hashtags based on commas
    hashtags_list = cleaned_str.split(", ")
    return hashtags_list

def main():
    # Get command-line arguments
    args = get_arguments()

    # If --all is set, override all other operations to be True
    if args.all:
        args.create_db = True
        args.create_schema = True
        args.create_warehouse = True
        args.create_table = True
        args.delete_table = True
    
    # Connect to Snowflake
    conn = snowflake.connector.connect(
        user=SNOWFLAKE_USER,
        password=SNOWFLAKE_PASSWORD,
        account=SNOWFLAKE_ACCOUNT,
        database=SNOWFLAKE_DATABASE,
        schema=SNOWFLAKE_SCHEMA,
        role=SNOWFLAKE_ROLE
    )
    
    with conn.cursor() as cur:
        try:
            # Create database if not exists
            if args.create_db:
                cur.execute(CREATE_DATABASE_SQL)
                print(f"Database '{SNOWFLAKE_DATABASE}' created or already exists.")

            # Create warehouse if not exists
            if args.create_warehouse:
                cur.execute(CREATE_WAREHOUSE_SQL)
                print(f"Warehouse '{SNOWFLAKE_WAREHOUSE}' created or replaced.")
                cur.execute(f"USE WAREHOUSE {SNOWFLAKE_WAREHOUSE};")

            # Create schema if not exists
            if args.create_schema:
                cur.execute(CREATE_SCHEMA_SQL)
                print(f"Schema '{SNOWFLAKE_SCHEMA}' created or already exists.")

            # Delete table if exists
            if args.delete_table:
                cur.execute(DELETE_TABLE_SQL)
                print(f"Table 'PREDICTED_TWEETS' deleted if it existed.")

            # Create table if not exists
            if args.create_table:
                cur.execute(CREATE_TABLE_SQL)
                print(f"Table 'PREDICTED_TWEETS' created or replaced.")

        except snowflake.connector.errors.ProgrammingError as e:
            print(f"Error during execution: {e}")
            raise
    
    # Insert DataFrame into Snowflake if CSV path is provided
    if args.csv_path:
        try:
            df = pd.read_csv(args.csv_path, parse_dates=['date_time'], dayfirst=True)
            df.columns = df.columns.str.upper()
            
            array_of_arrays = df['HASHTAGS_FINAL'].tolist()
            df['HASHTAGS_FINAL'] = df['HASHTAGS_FINAL'].apply(clean_hashtags)
            
            write_pandas(conn, df, 'PREDICTED_TWEETS')
            
            print(f"Successfully inserted")

        except KeyError as e:
            print(f"KeyError: {e}")
            print("Available columns in the CSV:", df.columns)
            raise
        except FileNotFoundError as e:
            print(f"FileNotFoundError: {e}")
            raise
    
    conn.close()

if __name__ == "__main__":
    main()
