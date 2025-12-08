from bs4 import BeautifulSoup
import requests
from dotenv import load_dotenv
import pandas as pd
import os
from datetime import date
import json

class ap_poll_manager:
    def __init__(self):
        self.env = load_dotenv()
        self.headers = {
            "User-Agent": "your browser UA string",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Referer": "https://example.com",
            }
        self.first_season_espn = os.environ.get('FIRST_SEASON_ESPN')
        self.last_season_espn = os.environ.get('CUR_SEASON_ESPN')
        # Manually add starting weeks for each poll
        with open('configs/week_starts.json', 'w') as file:
            self.week_starts = json.load(file)
        self.week_starts = {}

for year in range(int(os.environ.get('FIRST_SEASON_ESPN')), int(os.environ.get('CUR_SEASON_ESPN')) + 1):
    # loop through each while the website response is valid

    # 2006 season is missing week 2 for some reason? maybe add tolerance of one extra try
    invalid_count = 0
    for week in range(1, 22):
        response = requests.get(f'https://www.espn.com/mens-college-basketball/rankings/_/week/{week}/year/{year}/seasontype/2', headers=headers)
        html_content = response.text
        soup = BeautifulSoup(html_content, 'html.parser')
        divs = soup.find_all('div')
        last_week = False
        for d in divs:
            if 'No Data Available' in d.text:
                print(f'First invalid week for season {year} is {week}')
                invalid_count += 1
                break
        
        # Two error tolerance 
        if invalid_count == 1:
            continue
        elif invalid_count == 2:
            break

        df = pd.read_html(html_content, flavor="bs4")[0]

        trends = []
        trs = soup.find_all('tr')
        for tr in trs:
            for td in tr.find_all('td'):
                for div in td.find_all('div'):
                    cls = div.get('class')
                    trend = div.text
                    if 'positive' in cls:
                        trends.append(int(trend))
                    elif 'negative' in cls:
                        trends.append(-int(trend))
                    elif 'trend' in cls:
                        trends.append(0)

        # Find the "others receiving votes" text
        other_teams = []
        paragraphs = soup.find_all('p')
        for p in paragraphs[0:2]:
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
        df['Team'] = df['Team'].str.replace('\(\d+\)', '', regex=True).str.rstrip().str.split(' ').str[1:].str.join(' ')

        combined_df = pd.concat((df[['Team', 'PTS', 'first']], other_df))
        combined_df = combined_df.rename(columns = {'Team': 'team', 'PTS': 'votes', 'first': 'first_votes'})
        combined_df['rank'] = combined_df['votes'].rank(method='min', ascending=False).astype(int)
        combined_df.reset_index(drop=True, inplace=True)
        combined_df['week'] = week
        combined_df['season'] = year - 2000
        print(combined_df.head(50))
            



response = requests.get('https://www.espn.com/mens-college-basketball/rankings/_/week/9/year/2014/seasontype/2', headers=headers)
html_content = response.text
soup = BeautifulSoup(html_content, 'html.parser')

df = pd.read_html(html_content, flavor="bs4")[0]

trends = []
trs = soup.find_all('tr')
for tr in trs:
    for td in tr.find_all('td'):
        for div in td.find_all('div'):
            cls = div.get('class')
            trend = div.text
            if 'positive' in cls:
                trends.append(int(trend))
            elif 'negative' in cls:
                trends.append(-int(trend))
            elif 'trend' in cls:
                trends.append(0)

# Find the "others receiving votes" text
other_teams = []
paragraphs = soup.find_all('p')
for p in paragraphs[0:2]:
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

# Subset the first 25 (AP Poll only, everything after is coaches poll)
trends = trends[:25]

# Replace unsigned trends with positive-negative trends
#df['TREND'] = trends

# Find first place votes
df['first'] = df['Team'].str.findall('\((\d+)\)').str[0]
df['first'].fillna(0, inplace=True)

# Team abbreviations
#df['abv'] = df['Team'].str.split(' ').str[0]

# Reformatted team name
df['Team'] = df['Team'].str.replace('\(\d+\)', '', regex=True).str.rstrip().str.split(' ').str[1:].str.join(' ')

# Team wins
#df['wins'] = df['REC'].str.split('-').str[0].astype(int)

# Team losses
#df['losses'] = df['REC'].str.split('-').str[1].astype(int)
#df['season'] = season
#df['week'] = week

combined_df = pd.concat((df[['Team', 'PTS', 'first']], other_df))
combined_df = combined_df.rename(columns = {'Team': 'team', 'PTS': 'votes', 'first': 'first_votes'})
combined_df['rank'] = combined_df['votes'].rank(method='min', ascending=False).astype(int)
combined_df.reset_index(drop=True, inplace=True)
combined_df['week'] = week
combined_df['season'] = year - 2000
#df.rename(columns = {'RK': 'rank', 'Team': 'team', 'REC'})

#print(df.head(15))


#if not rows:
    # Last season was final season, continue here
    #pass
#print(rows)