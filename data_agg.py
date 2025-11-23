import requests
import csv
import os
import pandas as pd
from supabase import create_client
from dotenv import load_dotenv
from datetime import datetime
from pprint import pprint
import json 

os.makedirs('games', exist_ok = True)

load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY"))
cur_season = int(os.getenv("CUR_SEASON"))
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
    game = f'games/season_{season:02}.tsv'
    df = pd.read_fwf(game, names=columns, widths=None, engine='python', header=None, colspecs=colspecs)
    return df

def get_cleaned_game_df(season: int):
        # Extract the neutrality and overtimeof each game from the dataframe
        # This part represents home teams as 'team'. Must populate the opposite matchups after
    df = read_season(season)
    df['season'] = season
    df['overtime'] = df['neutral'].str.split().str[0].str.contains(pat='\d*',regex=True)
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
    json_model = json.loads(df.to_json(orient='records'))
    response = table.insert(json_model).execute()


    



if __name__ == '__main__':
    for i in range(1,cur_season+1):
        df = get_cleaned_game_df(i)
        insert_df_into_db('games', df)
    