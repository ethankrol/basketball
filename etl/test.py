# Test file to get all team names from polls, all team names from games, and find the non-intersected team names.
import requests
import os
import pandas as pd
from supabase import create_client
from dotenv import load_dotenv
from datetime import datetime
from pprint import pprint
import json 
from io import StringIO
from collections import Counter

load_dotenv()

supabase_url = os.environ.get('SUPABASE_URL')
if not supabase_url:
    raise KeyError('Invalid supabase url in environment')
supabase_key = os.environ.get('SUPABASE_SERVICE_KEY')
if not supabase_key:
    raise KeyError('Invalid supabase key in environment')

supabase = create_client(supabase_url, supabase_key)

games_table = supabase.table('games')
polls_table = supabase.table('polls')

games_teams = games_table.select('team').execute().data
polls_teams = polls_table.select('team').execute().data
spelling_table = supabase.table('team_spellings')
team_spellings = spelling_table.select('team_spelling').execute().data

games_set = set(k['team'] for k in games_teams)
polls_set = set(k['team'] for k in polls_teams)
spellings_set = set(k['team_spelling'] for k in team_spellings)

# Print out what doesnt exist between the two

all_teams = games_set | polls_set
non_present_teams = {s for s in all_teams if s not in spellings_set}
print(non_present_teams)