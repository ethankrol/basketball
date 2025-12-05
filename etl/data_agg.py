import requests
import os
import pandas as pd
from supabase import create_client
from dotenv import load_dotenv
from datetime import datetime
from pprint import pprint
import json 
from io import StringIO

load_dotenv()

supabase_url = os.environ.get('SUPABASE_URL')
if not supabase_url:
    raise KeyError('Invalid supabase url in environment')
supabase_key = os.environ.get('SUPABASE_SERVICE_KEY')
if not supabase_key:
    raise KeyError('Invalid supabase key in environment')

supabase = create_client(supabase_url, supabase_key)
columns = ['date', 'opponent', 'opponent_score', 'team', 'team_score', 'neutral']

# Colspecs to split the final misc later on 
colspecs = [(0,10), (11,33), (34,37), (38,61), (61,64),(65,90)]


def get_game_season_tsvs(start: int, end: int):
    #For current season, use start = 1, end = 27
    for season in range(start,end):
        games = requests.get(f'https://kenpom.com/cbbga{season:02}.txt')
        games_text = games.text
        
        with open(f'games/season_{season:02}.tsv', 'w', encoding = 'utf-8') as file:
            file.write(games_text)

def read_season(season: int) -> pd.DataFrame:
    season_request = requests.get(f'https://kenpom.com/cbbga{season:02}.txt')
    season_request.raise_for_status()
    season_text = season_request.text
    season_file = StringIO(season_text)
    if not season_file:
        raise ValueError("Error turning API request into text file")
    
    df = pd.read_fwf(season_file, names=columns, widths=None, engine='python', header=None, colspecs=colspecs)
    return df

def get_cleaned_game_df(season: int):
        # Extract the neutrality and overtimeof each game from the dataframe
        # This part represents home teams as 'team'. Must populate the opposite matchups after
    df = read_season(season)
    df['season'] = season
    df['overtime'] = df['neutral'].str.split().str[0].str.contains(pat='\d',regex=True, na=False)
    df['neutral'] = df['neutral'].str.split().str[0].str.contains('N', case=False, na=False)
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    df['home'] = True

    # Now we need to populate the games by swapping home and away teams.
    swap = df.copy()
    swap_cols = columns + ['season', 'overtime', 'home']
    swap_cols[1], swap_cols[3] = swap_cols[3], swap_cols[1]
    swap_cols[2], swap_cols[4] = swap_cols[4], swap_cols[2]
    swap.columns = swap_cols
    swap['home'] = False
    all_games = pd.concat([df, swap], axis=0)
    return all_games

def insert_df_into_db(table_name: str, df: pd.DataFrame):
    table = supabase.table(table_name)
    if not table:
        raise ValueError(f'Table {table_name} not in db')
    json_model = json.loads(df.to_json(orient='records'))
    response = table.insert(json_model).execute()
    return response

def upsert_df_into_db(table_name: str, df: pd.DataFrame):
    table = supabase.table(table_name)
    if not table:
        raise ValueError(f'Table {table_name} not in db')
    #json_model = json.loads(df.to_json(orient='records'))
    json_model = df.to_dict(orient='records')
    response = table.upsert(json_model).execute()
    return response
