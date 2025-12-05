from bs4 import BeautifulSoup
import requests
from dotenv import load_dotenv

import requests
import pandas as pd

# Instead of feeding into beautifulsoup to parse the table tags yourself,
# feed the html into pandas to do that for you 

load_dotenv()
headers = {
    "User-Agent": "your browser UA string",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://example.com",
}

for year in range(2003, 2027):
    pass

response = requests.get('https://www.espn.com/mens-college-basketball/rankings/_/week/9/year/2014/seasontype/2', headers=headers)
html_content = response.text

df = pd.read_html(html_content, flavor="bs4")[0]
print(df.head(5))

soup = BeautifulSoup(html_content, 'html.parser')
trends = []
trs = soup.find_all('tr')
for tr in trs:
    for td in tr.find_all('td'):
        for div in td.find_all('div'):
            print(div)
            cls = div.get('class')
            trend = div.text
            if 'positive' in cls:
                trends.append(int(trend))
            elif 'negative' in cls:
                trends.append(-int(trend))
            elif 'trend' in cls:
                trends.append(0)


# Subset the first 25 (AP Poll only, everything after is coaches poll)
trends = trends[:25]
print(trends)
df['TREND'] = trends
print(df.head(15))


#if not rows:
    # Last season was final season, continue here
    #pass
#print(rows)