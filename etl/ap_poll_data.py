from bs4 import BeautifulSoup
import requests
from dotenv import load_dotenv
import pandas as pd
import os

load_dotenv()
headers = {
    "User-Agent": "your browser UA string",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://example.com",
}

for year in range(int(os.environ.get('FIRST_SEASON_ESPN')), int(os.environ.get('CUR_SEASON_ESPN')) + 1):
    week = 1

response = requests.get('https://www.espn.com/mens-college-basketball/rankings/_/week/9/year/2014/seasontype/2', headers=headers)
html_content = response.text

df = pd.read_html(html_content, flavor="bs4")[0]

soup = BeautifulSoup(html_content, 'html.parser')
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
print(other_df)

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
print(combined_df.head(45))
#df.rename(columns = {'RK': 'rank', 'Team': 'team', 'REC'})

#print(df.head(15))


#if not rows:
    # Last season was final season, continue here
    #pass
#print(rows)