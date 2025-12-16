from bs4 import BeautifulSoup
import requests
from dotenv import load_dotenv
import pandas as pd
import os
from datetime import date, timedelta, datetime
import json
from supabase import create_client
from data_agg import upsert_df_into_db

class APPollManager:
    def __init__(self):
        load_dotenv()
        self.headers = {
            "User-Agent": "your browser UA string",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Referer": "https://example.com",
            }
        self.first_season_espn = int(os.environ.get('FIRST_SEASON_ESPN'))
        self.last_season_espn = int(os.environ.get('CUR_SEASON_ESPN'))
        self.cur_season = int(os.environ.get('CUR_SEASON'))

        # Manually add starting weeks for each poll
        with open('etl/configs/week_starts.json', 'r') as file:
            self.week_starts = json.load(file)
            self.week_starts = {int(k): datetime.strptime(v, '%Y-%m-%d').date() for (k,v) in self.week_starts.items()}

    def get_start_week(self, season: int) -> date:
        """Returns the season start date for a specific season (year). Seasons are represented by starting in the new year. Ex. 2025-2026 is represented by 2026.
            Raises a KeyError if the season is not found.
            """
        season_start = self.week_starts.get(season, -1)
        if season_start == -1:
            raise KeyError(f'Season {season} not found')
        
        return season_start
    
    def get_current_season(self) -> int:
        """Returns the current season (ex. 2026)"""
        today = datetime.now()
        month = today.month
        year = today.year
        if month > 10:
            year += 1

        return year
    
    def get_current_week(self) -> int:
        """Returns the current integer number of weeks from the start deate of the given season. Preseason is week 1. First poll is week 2."""
        season = self.get_current_season()
        start_date = self.get_start_week(season)
        today = datetime.now().date()
        if today < start_date:
            week = 1
        else:
            week = int((today - start_date).days) // 7 + 2

        return week

    def get_week_poll(self, week: int, year: int, invalid_count=0) -> pd.DataFrame:
        response = requests.get(f'https://www.espn.com/mens-college-basketball/rankings/_/week/{week}/year/{year}/seasontype/2', headers=self.headers)
        html_content = response.text
        soup = BeautifulSoup(html_content, 'html.parser')
        divs = soup.find_all('div')
        for d in divs:
            if 'No Data Available' in d.text:
                print(f'First invalid week for season {year} is {week}')
                invalid_count += 1
                raise FileNotFoundError('Season not found')

        df = pd.read_html(html_content, flavor="bs4")[0]

        # Find the "others receiving votes" text
        other_teams = []
        paragraphs = soup.find_all('p')
        for p in paragraphs[0:1]:
            span = p.find_all('span')
            for s in span:
                if "Others" in s.text:
                    teams_text = p.text.split(':')[1]
                    teams_split = teams_text.strip().split(', ')
                    for team in teams_split:
                        split_team_score = team.split(' ')
                        team = ' '.join(split_team_score[:-1])
                        score = int(split_team_score[-1])
                        other_teams.append((team, score, 0))

        other_teams.sort(key = lambda x : x[1], reverse=True)
        other_df = pd.DataFrame(other_teams, columns = ['Team', 'PTS', 'first'])

        df['first'] = df['Team'].str.findall('\((\d+)\)').str[0]
        df['first'].fillna(0, inplace=True)
        df['Team'] = df['Team'].str.replace('\(\d+\)', '', regex=True).str.rstrip().str.split(' ').str[1:].str.join(' ').str.lower()
        print(df.head())

        combined_df = pd.concat((df[['Team', 'PTS', 'first']], other_df))
        combined_df = combined_df.rename(columns = {'Team': 'team', 'PTS': 'votes', 'first': 'first_votes'})
        combined_df['first_votes'] = combined_df['first_votes'].astype(int)
        combined_df['rank'] = combined_df['votes'].rank(method='min', ascending=False).astype(int)
        combined_df.reset_index(drop=True, inplace=True)
        combined_df['week'] = week - invalid_count
        combined_df['season'] = year - 2000

        # Need to normalize vote totals
        vote_totals = combined_df['votes'].sum()
        total_points = (25*26)/2
        num_voters = vote_totals / total_points
        combined_df['normal_votes'] = combined_df['votes'].div(other=num_voters*25).clip(upper=1.0)

        # Normalize first vote totals
        first_vote_total = combined_df['first_votes'].sum() 
        combined_df['normal_first_votes'] = combined_df['first_votes'].div(other=first_vote_total).clip(upper=1.0)

        # Logic for date calculation. If it's the first week then we just fill in some random early date
        if week == 1:
            combined_df['date'] = date(day=1, month=10, year=year).isoformat()
        else:
            season_start = self.week_starts.get(year, -1)
            if season_start == -1:
                raise KeyError(f'Season start date for season {year} not found')
            cur_date = season_start + timedelta(days=((week-2-invalid_count)*7))
            combined_df['date'] = cur_date.isoformat()
        
        return combined_df
        


    def insert_all_polls(self):
        for year in range(self.first_season_espn, self.last_season_espn + 1):
            # loop through each while the website response is valid
            # 2006 season is missing week 2 for some reason? maybe add tolerance of one extra try

            invalid_count = 0
            for week in range(1, 22):
                # 2011 season starts one week later and has no preseason poll
                if year == 2011 and week == 1:
                    continue
                try:
                    df = self.get_week_poll(year=year, week=week, invalid_count=invalid_count)
                    res = upsert_df_into_db('polls', df)
                except FileNotFoundError as e:
                    print(f'Season {week} not found')
                    invalid_count +=1
                    continue

    def insert_week_poll(self, week: int, season: int):
        try:
            df = self.get_week_poll(year=season, week=week)
            res = upsert_df_into_db('polls', df)
            return True 
        except FileNotFoundError:
            print(f'Poll for season: {season}, week {week} could not be found. Nothing inserted')
            return False