import requests
import csv
import os
import pandas as pd
from supabase import create_client
from datetime import datetime
from dotenv import load_dotenv
import json 
from io import StringIO
from data_agg import upsert_df_into_db, get_cleaned_game_df

def get_current_season():
    today = datetime.now()
    month = int(today.month)
    year = int(str(today.year)[-2:])

    if month > 10:
        return year + 1
    return year 

if __name__ == '__main__':
    season = get_current_season()
    df = get_cleaned_game_df(season)
    res = upsert_df_into_db('games', df)
    


