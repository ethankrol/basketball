import requests
import csv
import os
import pandas as pd
from supabase import create_client
from datetime import datetime, date
from dotenv import load_dotenv
import json 
from io import StringIO
from data_agg import upsert_df_into_db
from ap_poll_data import APPollManager

def get_current_season():
    today = datetime.now()
    month = int(today.month)
    year = int(str(today.year)[-2:])

    if month > 10:
        return year + 1
    return year 

def get_current_week(season_start_date: date) -> int:
    """Given the start date of a season, return the current week as a date starting on Monday."""
    today = datetime.now().date()
    return(today)

def insert_current_week():
    ap = APPollManager()
    week = ap.get_current_week()
    season = ap.get_current_season()
    res = ap.insert_week_poll(week, season)
    if res:
        print(f'Successfully inserted season {season}, week {week} ap poll into database')
    else:
        print(f'Failure inserting season {season}, week {week} ap poll into database')
    

if __name__ == '__main__':
    insert_current_week()
    