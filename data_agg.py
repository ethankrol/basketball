import requests
import csv
import os
import pandas as pd

def get_game_season_tsvs():
    for season in range(1, 27):
        games = requests.get(f'https://kenpom.com/cbbga{season:02}.txt')
        games_text = games.text
        
        with open(f'season_{season:02}.tsv', 'w', encoding = 'utf-8') as file:
            file.write(games_text)

def read_season(season: int) -> pd.DataFrame:
    game = f'games/season_{season:02}.tsv'
    print(game)
    df = pd.read_csv(game, sep ='\t')
    print(df.head())
    return df


if __name__ == '__main__':
    for season in range(1,27):
        df = read_season(season)
    