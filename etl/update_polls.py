import pandas as pd
from datetime import datetime, date
from dotenv import load_dotenv
from data_agg import upsert_df_into_db
from ap_poll_data import APPollManager

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
    