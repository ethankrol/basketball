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

def get_current_week():
    today = datetime.now().date()
    
    

if __name__ == '__main__':
    ap = APPollManager()
    poll = ap.get_week_poll(1, 2026)
    ap.insert_all_polls()
    