import pandas as pd
from datetime import datetime, date
from dotenv import load_dotenv
from data_agg import upsert_df_into_db, get_data, get_db
from ap_poll_data import APPollManager
import json

load_dotenv()

class EloPipeline:
    def __init__(self, initial_data):
        self.state = {}

        self.team_states = get_data('team_id, conference_2001')
        print(self.team_states)

teams = get_data('*', 'teams')
team_map = get_data('*', 'team_spellings')

state = {}

def add_conferences():
    with open('data/conferences_2001.json', 'r') as f:
        raw_data = json.load(f)
    upsert_data = [
        {
            "team_id": d["team_id"], 
            "conference_2001": d["conference_2001"]
        } 
        for d in raw_data
    ]
    supabase = get_db()
    response = supabase.table('teams').upsert(upsert_data).execute()

#def add_elo(year: int):

if __name__ == '__main__':
    print('hi')
    pipeline = EloPipeline(2)
